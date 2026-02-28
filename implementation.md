import json
import os
import re
import time
import traceback
import threading
import atexit
import gc
import asyncio
import concurrent.futures
from typing import Any, Optional, Dict, List, Union, Tuple
from functools import lru_cache
import weakref
import logging

import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings

# Optional: use InstructorEmbedding package directly to avoid SentenceTransformer fallback warnings
try:
    from InstructorEmbedding import INSTRUCTOR as _INSTRUCTOR_CLASS
except ImportError:
    _INSTRUCTOR_CLASS = None
from contextlib import contextmanager, asynccontextmanager

import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

import sys
import torch
import torch.nn.functional as F

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__),'../'))
sys.path.append(project_root)

from knowledgebase.content_classifier import get_content_classifier, ContentClassifier

# Environment detection: production when running under /var/www (e.g. intgr8_api_server)
IS_PRODUCTION = os.path.exists("/var/www")
IS_DEVELOPMENT = not IS_PRODUCTION

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Embedding backend: "instructor" (CPU) or "sentence_transformer".
# Override via env: RAG_EMBEDDING_BACKEND=instructor|sentence_transformer
# Or at runtime: GlobalEmbeddingManager.EMBEDDING_BACKEND = "sentence_transformer" (before first use).
EMBEDDING_BACKEND = os.environ.get("RAG_EMBEDDING_BACKEND", "instructor").strip().lower()
if EMBEDDING_BACKEND not in ("instructor", "sentence_transformer"):
    EMBEDDING_BACKEND = "instructor"
logger.info("RAG embedding backend: %s", EMBEDDING_BACKEND)


# Default retrieval instruction for Instructor (used for both docs and queries when Chroma calls us)
_INSTRUCTOR_RETRIEVAL_INSTRUCTION = "Represent the following text for retrieval:"


class InstructorChromaEmbeddingFunction:
    """
    Custom Chroma embedding function using the InstructorEmbedding package's INSTRUCTOR
    model directly (not via Chroma's InstructorEmbeddingFunction). This avoids
    SentenceTransformer loading the model as a generic model with mean pooling and
    related warnings. Uses instruction-based encoding for retrieval.
    """

    def __init__(
        self,
        model_name: str = "hkunlp/instructor-base",
        device: str = "cpu",
        instruction: str = _INSTRUCTOR_RETRIEVAL_INSTRUCTION,
    ):
        self._model_name = model_name
        self._device = device
        self._instruction = instruction
        self._model = None
        self._lock = threading.Lock()

    @classmethod
    def name(cls) -> str:
        """Chroma compatibility: returns identifier used in config and registry."""
        return "instructor-chroma"

    def is_legacy(self) -> bool:
        """Chroma compatibility: non-legacy so config is serialized/deserialized by name."""
        return False

    def default_space(self) -> str:
        """Chroma compatibility: default distance space (we use cosine)."""
        return "cosine"

    def supported_spaces(self) -> List[str]:
        """Chroma compatibility: supported distance spaces."""
        return ["cosine", "l2", "ip"]

    def get_config(self) -> Dict[str, Any]:
        """Chroma compatibility: return serializable config."""
        return {
            "model_name": self._model_name,
            "device": self._device,
            "instruction": self._instruction,
        }

    def validate_config(self, config: Dict[str, Any]) -> None:
        """Chroma compatibility: validate config (no-op)."""
        pass

    def validate_config_update(
        self, old_config: Dict[str, Any], new_config: Dict[str, Any]
    ) -> None:
        """Chroma compatibility: validate config update (no-op)."""
        pass

    @classmethod
    def build_from_config(cls, config: Dict[str, Any]) -> "InstructorChromaEmbeddingFunction":
        """Chroma compatibility: build instance from config (used when loading collection)."""
        return cls(
            model_name=config.get("model_name", "hkunlp/instructor-base"),
            device=config.get("device", "cpu"),
            instruction=config.get("instruction", _INSTRUCTOR_RETRIEVAL_INSTRUCTION),
        )

    def _ensure_model(self):
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if _INSTRUCTOR_CLASS is None:
                raise RuntimeError(
                    "InstructorEmbedding package is not installed; install it or use RAG_EMBEDDING_BACKEND=sentence_transformer"
                )
            self._model = _INSTRUCTOR_CLASS(self._model_name)
            self._model.to(self._device)
            logger.info("InstructorChromaEmbeddingFunction: loaded INSTRUCTOR model %s on %s", self._model_name, self._device)

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Chroma interface: embed a list of text strings. Returns list of embedding vectors."""
        if not input:
            return []
        self._ensure_model()
        # Instructor expects list of [instruction, text] pairs
        pairs = [[self._instruction, text] for text in input]
        embeddings = self._model.encode(
            pairs,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()
        return [emb.tolist() if hasattr(emb, "tolist") else list(emb) for emb in embeddings]

    def embed_query(
        self,
        input: Union[str, List[str], None] = None,
        *,
        query: Union[str, List[str], None] = None,
    ) -> Union[List[float], List[List[float]]]:
        """Chroma query path: embed query string(s). Chroma may call with keyword 'input' or 'query'."""
        texts = input if input is not None else query
        if texts is None:
            return []
        if isinstance(texts, str):
            result = self([texts])
            return result[0] if result else []
        return self(list(texts)) if texts else []


# Register so Chroma can rebuild this embedding function from config when loading collections
try:
    register_embedding_function = getattr(
        embedding_functions, "register_embedding_function", None
    )
    if register_embedding_function is not None:
        register_embedding_function(InstructorChromaEmbeddingFunction)
except Exception as e:
    logger.debug("Could not register InstructorChromaEmbeddingFunction with Chroma: %s", e)


# Global application-level singletons with proper thread safety
class GlobalEmbeddingManager:
    """Thread-safe singleton for managing embedding models across the application."""

    # Class-level config: switch embedding backend without editing _initialize_models.
    # Options: "instructor" (InstructorEmbedding, CPU), "sentence_transformer" (all-mpnet-base-v2).
    EMBEDDING_BACKEND = EMBEDDING_BACKEND

    _instance = None
    _lock = threading.RLock()
    _embedding_function = None
    _reranker = None
    _embedding_model = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(GlobalEmbeddingManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    if IS_PRODUCTION:
                        logger.info("[PRODUCTION] Loading RAG embedding models at startup...")
                        self._initialize_models()
                    else:
                        logger.info("[DEVELOPMENT] RAG embedding models will load on first access")
                    self._initialized = True

    def _initialize_models(self):
        """Initialize embedding models with error handling and retries. Backend selected by EMBEDDING_BACKEND."""
        max_retries = 3
        env_type = "PRODUCTION" if IS_PRODUCTION else "DEVELOPMENT"
        backend = getattr(self.__class__, "EMBEDDING_BACKEND", EMBEDDING_BACKEND)

        for attempt in range(max_retries):
            try:
                logger.info("Initializing RAG embedding models (%s) in %s mode (attempt %s/%s)", backend, env_type, attempt + 1, max_retries)

                if self._embedding_function is None:
                    if backend == "instructor":
                        if _INSTRUCTOR_CLASS is not None:
                            try:
                                self._embedding_function = InstructorChromaEmbeddingFunction(
                                    model_name="hkunlp/instructor-base",
                                    device="cpu",
                                )
                                logger.info("Instructor embedding initialized via InstructorEmbedding package (CPU)")
                            except Exception as e:
                                logger.warning(f"InstructorChromaEmbeddingFunction failed: {e}, falling back to SentenceTransformer")
                                self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                                    model_name="all-mpnet-base-v2",
                                    device="cpu",
                                )
                                logger.info("SentenceTransformer embedding function initialized (fallback)")
                        else:
                            logger.warning("InstructorEmbedding package not installed, using SentenceTransformer")
                            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                                model_name="all-mpnet-base-v2",
                                device="cpu",
                            )
                            logger.info("SentenceTransformer embedding function initialized (fallback)")
                    else:
                        self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                            model_name="all-mpnet-base-v2",
                            device="cuda" if torch.cuda.is_available() else "cpu",
                        )
                        logger.info("SentenceTransformer embedding function initialized")

                if self._reranker is None:
                    try:
                        self._reranker = CrossEncoder("cross-encoder/stsb-roberta-base")
                        logger.info("CrossEncoder reranker initialized")
                    except Exception as e:
                        logger.warning(f"Could not initialize reranker: {e}")
                        self._reranker = None

                if self._embedding_model is None and backend == "sentence_transformer":
                    try:
                        self._embedding_model = SentenceTransformer("all-mpnet-base-v2")
                        logger.info("Base SentenceTransformer model initialized")
                    except Exception as e:
                        logger.warning(f"Could not initialize base embedding model: {e}")
                        self._embedding_model = None

                logger.info("All RAG embedding models initialized successfully in %s mode", env_type)
                break

            except Exception as e:
                logger.error(f"Failed to initialize embedding models on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    logger.error("Failed to initialize embedding models after all retries")
                else:
                    time.sleep(2 ** attempt)
    
    def get_embedding_function(self):
        """Get the embedding function with environment-aware lazy loading."""
        # In development mode, check if models need to be loaded lazily
        if IS_DEVELOPMENT and self._embedding_function is None:
            with self._lock:
                if self._embedding_function is None:
                    logger.info("[DEVELOPMENT] Loading embedding models on first access...")
                    self._initialize_models()
        
        # In production mode, models should already be loaded, but ensure initialization
        elif IS_PRODUCTION and self._embedding_function is None:
            with self._lock:
                if self._embedding_function is None:
                    logger.warning("Production embedding models not initialized, loading now...")
                    self._initialize_models()
        
        return self._embedding_function
    
    def get_reranker(self):
        """Get the reranker with environment-aware lazy loading."""
        # In development mode, check if models need to be loaded lazily
        if IS_DEVELOPMENT and self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    logger.info("[DEVELOPMENT] Loading reranker model on first access...")
                    self._initialize_models()
        
        # In production mode, models should already be loaded, but ensure initialization
        elif IS_PRODUCTION and self._reranker is None:
            with self._lock:
                if self._reranker is None:
                    logger.warning("Production reranker model not initialized, loading now...")
                    self._initialize_models()
        
        return self._reranker
    
    def get_embedding_model(self):
        """Get the base embedding model with environment-aware lazy loading."""
        # In development mode, check if models need to be loaded lazily
        if IS_DEVELOPMENT and self._embedding_model is None:
            with self._lock:
                if self._embedding_model is None:
                    logger.info("[DEVELOPMENT] Loading base embedding model on first access...")
                    self._initialize_models()
        
        # In production mode, models should already be loaded, but ensure initialization
        elif IS_PRODUCTION and self._embedding_model is None:
            with self._lock:
                if self._embedding_model is None:
                    logger.warning("Production base embedding model not initialized, loading now...")
                    self._initialize_models()
        
        return self._embedding_model
    
    def health_check(self):
        """Check the health of embedding models."""
        return {
            "embedding_function": self._embedding_function is not None,
            "reranker": self._reranker is not None,
            "embedding_model": self._embedding_model is not None,
            "cuda_available": torch.cuda.is_available()
        }

# Global instance
_global_embedding_manager = GlobalEmbeddingManager()

# Thread-safe RAG manager registry
class RAGManagerRegistry:
    """Thread-safe registry for RAG manager instances."""
    
    def __init__(self):
        self._managers: Dict[str, 'RAGManager'] = {}
        self._lock = threading.RLock()
        self._weak_refs: Dict[str, weakref.ref] = {}
    
    def get_manager(self, path: str) -> 'RAGManager':
        """Get or create a RAG manager for the given path."""
        with self._lock:
            if path in self._managers:
                return self._managers[path]
            
            # Check weak references for cleaned up instances
            if path in self._weak_refs:
                manager = self._weak_refs[path]()
                if manager is not None:
                    self._managers[path] = manager
                    return manager
                else:
                    del self._weak_refs[path]
            
            # Create new manager
            manager = RAGManager._create_instance(path)
            self._managers[path] = manager
            
            # Set up weak reference for cleanup
            def cleanup_callback(ref):
                with self._lock:
                    if path in self._weak_refs and self._weak_refs[path] is ref:
                        del self._weak_refs[path]
                    if path in self._managers:
                        del self._managers[path]
            
            self._weak_refs[path] = weakref.ref(manager, cleanup_callback)
            return manager
    
    def cleanup_manager(self, path: str):
        """Clean up a specific manager."""
        with self._lock:
            if path in self._managers:
                manager = self._managers[path]
                manager._force_cleanup()
                del self._managers[path]
            if path in self._weak_refs:
                del self._weak_refs[path]
    
    def cleanup_all(self):
        """Clean up all managers."""
        with self._lock:
            for manager in list(self._managers.values()):
                manager._force_cleanup()
            self._managers.clear()
            self._weak_refs.clear()

# Global registry
_rag_registry = RAGManagerRegistry()

# Thread pool for async operations
_thread_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, (os.cpu_count() or 1) + 4),
    thread_name_prefix="RAGManager"
)

# Batch size for Chroma add (larger = fewer round-trips; embedding runs in batches)
RAG_ADD_BATCH_SIZE = int(os.environ.get("RAG_ADD_BATCH_SIZE", "32"))

# Dedicated pool for parallel per-chunk NLP (keywords, summary, key_facts)
# Smaller than main pool to avoid overwhelming the content classifier
_NLP_POOL_MAX_WORKERS = int(os.environ.get("RAG_NLP_WORKERS", "4"))
_nlp_prepare_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=max(1, min(_NLP_POOL_MAX_WORKERS, (os.cpu_count() or 1))),
    thread_name_prefix="RAGNLP"
)



# Legacy preset keys (tokens); prefer absolute chunk_size + overlap for API
CHUNK_SIZES = {
    "a2": [512*2, 128*2],
    "a3": [1024*2, 256*2],
    "a4": [2048*2, 512*2],
    "a5": [3072*2, 768*2],
    "a6": [4096*2, 1024*2],
    "a7": [6144*2, 1536*2],
    "a8": [8192*2, 2048*2],
    "all": [10000000*2, 1000*2],
}

# Overlap (characters) per chunk_size band for RAG; used when overlap not explicitly provided
# Band (min_chars, max_chars) -> overlap. chunk_size in [1000,2000) -> 128, [2000,4000) -> 256, etc.
OVERLAP_BAND_STEP = 2000
OVERLAP_PER_BAND = 128


def get_overlap_for_chunk_size(chunk_size: int) -> int:
    """Recommended overlap in characters for a given chunk_size (e.g. 1000–2000 -> 128, 2000–4000 -> 256)."""
    if chunk_size < 1000:
        return 128
    n = max(1, (chunk_size + OVERLAP_BAND_STEP - 1) // OVERLAP_BAND_STEP)
    return n * OVERLAP_PER_BAND

CONTEXT_LENGTH = {
    "500":500,
    "1000":1000,
    "1500":1500,
    "2000":2000,
    "2500":2500,
    "3000":3000,
    "4000":4000,
    "5000":5000,
    "7000":7000,
    "10000":10000,
    "15000":15000,
    "20000":20000,
    "30000":30000,
    "50000":50000
}

class RAGManager:
    """Production-ready, thread-safe RAG manager with connection pooling and failsafe mechanisms."""
    
    table_name = "rag_data"
    collection_name = "rag_collection"
    image_collection_name = "rag_images"

    def __init__(self, path: str):
        """Initialize RAG manager with path. Use get_instance() for singleton access."""
        self.DB_PATH = os.path.join(path, self.table_name)
        self.path = path
        self.client = None
        self.collection = None
        self.image_collection = None
        self._lock = threading.RLock()
        self._connection_pool = {}
        self._last_health_check = 0
        self._health_check_interval = 30  # seconds
        self._initialized = False
        self._initializing = False
        
        logger.info(f"RAGManager created for path: {self.DB_PATH}")

    @classmethod
    def get_instance(cls, path: str) -> 'RAGManager':
        """Get singleton instance for the given path."""
        return _rag_registry.get_manager(path)

    @classmethod
    def _create_instance(cls, path: str) -> 'RAGManager':
        """Create new instance (called by registry)."""
        instance = cls.__new__(cls)
        instance.__init__(path)
        atexit.register(instance._cleanup_on_exit)
        return instance

    def _ensure_initialized(self):
        """Ensure the manager is properly initialized with thread safety."""
        if self._initialized:
            return True
        
        with self._lock:
            if self._initialized:
                return True
                
            if self._initializing:
                # Wait for initialization to complete
                while self._initializing and not self._initialized:
                    time.sleep(0.1)
                return self._initialized
            
            self._initializing = True
            
            try:
                success = self._initialize_components()
                self._initialized = success
                return success
            except Exception as e:
                logger.error(f"Failed to initialize RAG manager: {e}")
                return False
            finally:
                self._initializing = False

    def _initialize_components(self) -> bool:
        """Initialize all components with error handling."""
        try:
            # Initialize client
            if not self._initialize_client():
                return False
            
            logger.info(f"RAGManager successfully initialized at {self.DB_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing RAG manager components: {e}")
            return False

    def _health_check(self) -> bool:
        """Perform health check on connections and models."""
        current_time = time.time()
        
        # Only perform health check if interval has passed
        if current_time - self._last_health_check < self._health_check_interval:
            return True
        
        with self._lock:
            try:
                # Check global embedding manager health
                embedding_health = _global_embedding_manager.health_check()
                
                # Check client connection
                client_healthy = True
                if self.client:
                    try:
                        # Simple health check - try to list collections
                        self.client.list_collections()
                    except Exception as e:
                        logger.warning(f"Client health check failed: {e}")
                        client_healthy = False
                        # Try to reinitialize client
                        self._initialize_client()
                
                # Check collections
                collections_healthy = True
                if client_healthy and self.client:
                    try:
                        if self.collection:
                            self.collection.count()
                        if self.image_collection:
                            self.image_collection.count()
                    except Exception as e:
                        logger.warning(f"Collections health check failed: {e}")
                        collections_healthy = False
                        # Reinitialize collections
                        self._initialize_collections()
                
                self._last_health_check = current_time
                overall_health = embedding_health["embedding_function"] and client_healthy and collections_healthy
                
                if not overall_health:
                    logger.warning("RAG manager health check failed, attempting recovery")
                
                return overall_health
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                return False

    def _initialize_client(self) -> bool:
        """Initialize client with proper error handling and retries."""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Initializing ChromaDB client (attempt {attempt + 1}/{max_retries})")
                
                # Close any existing connections first
                self._force_cleanup()
                
                # Small delay to ensure cleanup is complete
                if attempt > 0:
                    time.sleep(0.5 * attempt)
                
                # Ensure the directory exists
                os.makedirs(os.path.dirname(self.DB_PATH), exist_ok=True)
                
                # Initialize client with connection pooling settings
                self.client = chromadb.PersistentClient(
                    path=self.DB_PATH,
                    settings=Settings(
                        allow_reset=True,
                        is_persistent=True,
                        anonymized_telemetry=False  # Disable telemetry for production
                    ),
                    tenant=DEFAULT_TENANT,
                    database=DEFAULT_DATABASE,
                )
                
                # Initialize collections
                if not self._initialize_collections():
                    raise Exception("Failed to initialize collections")
                
                logger.info("ChromaDB client and collections initialized successfully")
                return True
            
            except Exception as e:
                logger.error(f"Error initializing client on attempt {attempt + 1}: {e}")
                self._force_cleanup()
                
                if attempt == max_retries - 1:
                    logger.error("Failed to initialize client after all retries")
                    return False
                else:
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False

    def _initialize_collections(self) -> bool:
        """Initialize collections with proper error handling."""
        try:
            if not self.client:
                logger.error("Cannot initialize collections: client is None")
                return False
            
            # Get embedding function from global manager
            embedding_function = _global_embedding_manager.get_embedding_function()
            if not embedding_function:
                logger.error("Cannot initialize collections: embedding function is None")
                return False
            
            # Create text collection
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name, 
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Text collection '{self.collection_name}' initialized")
            except Exception as e:
                logger.error(f"Failed to initialize text collection: {e}")
                return False
            
            # Create image collection
            try:
                self.image_collection = self.client.get_or_create_collection(
                    name=self.image_collection_name,
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                logger.info(f"Image collection '{self.image_collection_name}' initialized")
            except Exception as e:
                logger.error(f"Failed to initialize image collection: {e}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing collections: {e}")
            return False

    def _force_cleanup(self):
        """Force cleanup of all resources with proper error handling."""
        try:
            with self._lock:
                # Reset collections
                self.collection = None
                self.image_collection = None
                
            # Close client connections
            if self.client:
                try:
                    if hasattr(self.client, 'clear_system_cache'):
                        self.client.clear_system_cache()
                except Exception as e:
                    logger.debug(f"Error clearing client cache: {e}")
                finally:
                    self.client = None
                
                    # Clear connection pool
                    self._connection_pool.clear()
                
            # Force garbage collection
                gc.collect()
            
                logger.debug(f"Cleanup completed for RAG manager at {self.DB_PATH}")
                
        except Exception as e:
            logger.warning(f"Error during force cleanup: {e}")

    def _cleanup_on_exit(self):
        """Cleanup function to run on program exit."""
        self._force_cleanup()

    def close(self):
        """Proper cleanup method for external use."""
        self._force_cleanup()
        
    def close_all_connections(self):
        """Close all connections properly."""
        self._force_cleanup()

    def __del__(self):
        """Destructor with proper cleanup."""
        try:
            self._force_cleanup()
        except:
            pass

    # ─────────────────────────────────────────────
    # NLP METADATA EXTRACTION (Content Classifier)
    # ─────────────────────────────────────────────

    def _get_classifier(self) -> Optional[ContentClassifier]:
        """Lazy-load the content classifier singleton."""
        try:
            return get_content_classifier()
        except Exception as e:
            logger.warning(f"Content classifier unavailable: {e}")
            return None

    def extract_content_metadata(self, text: str, url: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the NLP content classifier on *text* and return a flat dict
        of scalar values suitable for ChromaDB metadata storage, plus a
        ``nlp_meta_json`` key containing the full rich analysis as a JSON
        string for structured retrieval.

        ChromaDB only supports str | int | float | bool metadata values,
        so lists are joined as comma-separated strings and nested dicts
        are flattened with underscore-separated keys.
        """
        classifier = self._get_classifier()
        if classifier is None:
            return {}

        try:
            analysis = classifier.analyze(text, url=url)
            if not isinstance(analysis, dict):
                return {}
            flat = self._flatten_classifier_output(analysis)
            # Store the full rich metadata as a compact JSON string for retrieval.
            # Limit large text fields to keep the payload manageable.
            analysis_for_storage = dict(analysis)
            if "summary" in analysis_for_storage:
                analysis_for_storage["summary"] = (analysis_for_storage.get("summary") or "")[:500]
            if "key_facts" in analysis_for_storage:
                analysis_for_storage["key_facts"] = (analysis_for_storage.get("key_facts") or [])[:10]
            try:
                flat["nlp_meta_json"] = json.dumps(
                    analysis_for_storage, ensure_ascii=False, default=str
                )[:12000]
            except Exception:
                pass
            return flat
        except Exception as e:
            logger.warning(f"Content metadata extraction failed: {e}")
            return {}

    @staticmethod
    def _flatten_classifier_output(analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert the rich classifier dict into flat scalars for ChromaDB filtering.

        Stored keys (all prefixed with ``nlp_``):
          subject, subject_confidence,
          document_type, document_type_confidence, document_types,
          language,
          sentiment, polarity, subjectivity,
          reading_level, word_count, vocabulary_richness,
          information_density, avg_sentence_length,
          keywords, topics, entities, named_concepts,
          summary, key_facts, temporal_markers,
          has_code, has_lists, has_tables, has_headers, paragraph_count,
          content_fingerprint
        """
        flat: Dict[str, Any] = {}

        # Subject / domain
        flat["nlp_subject"] = str(analysis.get("subject", ""))
        flat["nlp_subject_confidence"] = float(analysis.get("subject_confidence", 0.0))

        # Primary document type
        doc_type = analysis.get("document_type")
        if isinstance(doc_type, dict):
            flat["nlp_document_type"] = str(doc_type.get("type", ""))
            flat["nlp_document_type_confidence"] = float(doc_type.get("confidence", 0.0))
        else:
            flat["nlp_document_type"] = str(doc_type) if doc_type else ""
            flat["nlp_document_type_confidence"] = 0.0

        # Multi-label document types as "type:confidence" pairs
        doc_types = analysis.get("document_types", [])
        if isinstance(doc_types, list) and doc_types:
            flat["nlp_document_types"] = ", ".join(
                f"{dt.get('type', '')}:{dt.get('confidence', 0):.2f}"
                for dt in doc_types[:6]
                if isinstance(dt, dict)
            )

        # Language
        flat["nlp_language"] = str(analysis.get("language", ""))

        # Sentiment
        sentiment = analysis.get("sentiment")
        if isinstance(sentiment, dict):
            flat["nlp_sentiment"] = str(sentiment.get("sentiment", "neutral"))
            flat["nlp_polarity"] = float(sentiment.get("polarity", 0.0))
            flat["nlp_subjectivity"] = float(sentiment.get("subjectivity", 0.0))
        else:
            flat["nlp_sentiment"] = "neutral"
            flat["nlp_polarity"] = 0.0
            flat["nlp_subjectivity"] = 0.0

        # Readability
        readability = analysis.get("readability")
        if isinstance(readability, dict):
            flat["nlp_reading_level"] = str(readability.get("reading_level", ""))
            flat["nlp_word_count"] = int(readability.get("word_count", 0))
            flat["nlp_vocabulary_richness"] = float(readability.get("vocabulary_richness", 0.0))
            flat["nlp_information_density"] = float(readability.get("information_density", 0.0))
            flat["nlp_avg_sentence_length"] = float(readability.get("avg_sentence_length", 0.0))

        # Keywords
        keywords = analysis.get("keywords", [])
        if isinstance(keywords, list):
            flat["nlp_keywords"] = ", ".join(str(k) for k in keywords[:15])

        # Topics
        topics = analysis.get("topics_themes", [])
        if isinstance(topics, list):
            flat["nlp_topics"] = ", ".join(str(t) for t in topics[:10])

        # Named entities
        entities = analysis.get("entities", {})
        if isinstance(entities, dict):
            entity_texts = []
            for ent_type, ent_list in entities.items():
                for item in (ent_list or []):
                    name = item.get("text", item) if isinstance(item, dict) else str(item)
                    entity_texts.append(f"{ent_type}:{name}")
            flat["nlp_entities"] = ", ".join(entity_texts[:20])

        # Named concepts (noun phrases)
        named_concepts = analysis.get("named_concepts", [])
        if isinstance(named_concepts, list):
            flat["nlp_named_concepts"] = ", ".join(str(c) for c in named_concepts[:12])

        # Temporal markers
        temporal = analysis.get("temporal_markers", [])
        if isinstance(temporal, list):
            flat["nlp_temporal_markers"] = ", ".join(str(t) for t in temporal[:10])

        # Summary and key facts
        summary = analysis.get("summary", "")
        if summary:
            flat["nlp_summary"] = str(summary)[:500]

        facts = analysis.get("key_facts", [])
        if isinstance(facts, list) and facts:
            flat["nlp_key_facts"] = " | ".join(str(f) for f in facts[:8])

        # Structural features (boolean flags + counts for filtering)
        struct = analysis.get("structural_features")
        if isinstance(struct, dict):
            flat["nlp_has_code"] = bool(struct.get("has_code", False))
            flat["nlp_has_lists"] = bool(struct.get("has_lists", False))
            flat["nlp_has_tables"] = bool(struct.get("has_tables", False))
            flat["nlp_has_headers"] = bool(struct.get("has_headers", False))
            flat["nlp_has_questions"] = bool(struct.get("has_questions", False))
            flat["nlp_paragraph_count"] = int(struct.get("paragraph_count", 0))

        # Content fingerprint for deduplication
        fp = analysis.get("content_fingerprint", "")
        if fp:
            flat["nlp_content_fingerprint"] = str(fp)

        return flat

    @staticmethod
    def _build_metadata_where_clause(
        subject: Optional[str] = None,
        document_type: Optional[str] = None,
        language: Optional[str] = None,
        sentiment: Optional[str] = None,
        reading_level: Optional[str] = None,
        min_word_count: Optional[int] = None,
        keyword_contains: Optional[str] = None,
        has_code: Optional[bool] = None,
        has_tables: Optional[bool] = None,
        has_lists: Optional[bool] = None,
        temporal_contains: Optional[str] = None,
        content_fingerprint: Optional[str] = None,
        extra_where: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """
        Build a ChromaDB *where* filter dict from high-level metadata params.

        ChromaDB supports $and / $or / $eq / $ne / $gt / $gte / $lt / $lte / $contains.
        Returns None if no filters are specified (skip filtering).

        Extended params (all optional):
          has_code:           filter documents that contain code blocks
          has_tables:         filter documents that contain tables
          has_lists:          filter documents that contain bullet/numbered lists
          temporal_contains:  substring match within nlp_temporal_markers field
          content_fingerprint: exact match on deduplication fingerprint
        """
        conditions: List[Dict] = []

        if subject:
            conditions.append({"nlp_subject": {"$eq": subject}})
        if document_type:
            conditions.append({"nlp_document_type": {"$eq": document_type}})
        if language:
            conditions.append({"nlp_language": {"$eq": language}})
        if sentiment:
            conditions.append({"nlp_sentiment": {"$eq": sentiment}})
        if reading_level:
            conditions.append({"nlp_reading_level": {"$eq": reading_level}})
        if min_word_count is not None:
            conditions.append({"nlp_word_count": {"$gte": min_word_count}})
        if keyword_contains:
            conditions.append({"nlp_keywords": {"$contains": keyword_contains}})
        if has_code is not None:
            conditions.append({"nlp_has_code": {"$eq": has_code}})
        if has_tables is not None:
            conditions.append({"nlp_has_tables": {"$eq": has_tables}})
        if has_lists is not None:
            conditions.append({"nlp_has_lists": {"$eq": has_lists}})
        if temporal_contains:
            conditions.append({"nlp_temporal_markers": {"$contains": temporal_contains}})
        if content_fingerprint:
            conditions.append({"nlp_content_fingerprint": {"$eq": content_fingerprint}})

        if extra_where and isinstance(extra_where, dict):
            conditions.append(extra_where)

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # Async methods for better performance
    async def addDocumentToRagAsync(self, file_name: str, document: str, metadata: Optional[Dict] = None) -> bool:
        """Async version of document addition with better performance."""
        loop = asyncio.get_event_loop()
        
        def _add_document():
            return self.adDocumentToRag(file_name, document, metadata)
        
        return await loop.run_in_executor(_thread_pool, _add_document)

    async def addFileToRagAsync(
        self,
        file_name: str,
        content: str,
        chunk_size_key: str,
        metadata: Optional[Dict] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> bool:
        """
        Async version of file addition. Runs the sync addFileToRag in a thread
        so it benefits from parallel NLP and batch Chroma add without blocking
        the event loop.
        """
        if not self._ensure_initialized():
            logger.error("RAG manager not initialized")
            return False
        if not self._health_check():
            logger.warning("Health check failed, attempting recovery")
            if not self._ensure_initialized():
                return False
        content = str(content)
        if len(content) < 10:
            return False
        loop = asyncio.get_event_loop()

        def _run_add():
            return self.addFileToRag(
                file_name,
                content,
                chunk_size_key=chunk_size_key,
                metadata=metadata,
                chunk_size=chunk_size,
                overlap=overlap,
                extract_nlp_metadata=True,
            )

        return await loop.run_in_executor(_thread_pool, _run_add)

    async def queryCollectionAsync(
        self,
        query: str,
        keywords: List[str] = None,
        limit: int = 10,
        min_relevance_threshold: float = 0.55,
        subject: Optional[str] = None,
        document_type: Optional[str] = None,
        language: Optional[str] = None,
        sentiment: Optional[str] = None,
        reading_level: Optional[str] = None,
        min_word_count: Optional[int] = None,
        keyword_contains: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
        show_meta: bool = True,
    ) -> List[Dict]:
        """Async version of collection querying with metadata filtering.

        Args:
            show_meta: When True (default), results include ``meta_data`` (parsed
                       rich NLP metadata) and the full ``metadata`` dict.
                       When False, only ``doc``, ``content``, ``filename``, and
                       ``score`` are returned.
        """
        loop = asyncio.get_event_loop()

        def _query():
            return self.queryCollection(
                query,
                keywords or [],
                limit,
                min_relevance_threshold=min_relevance_threshold,
                subject=subject,
                document_type=document_type,
                language=language,
                sentiment=sentiment,
                reading_level=reading_level,
                min_word_count=min_word_count,
                keyword_contains=keyword_contains,
                metadata_filter=metadata_filter,
                show_meta=show_meta,
            )

        return await loop.run_in_executor(_thread_pool, _query)

   
    # Legacy method maintained for compatibility
    def create_client(self):
        """Legacy method - use _initialize_client instead."""
        return self._initialize_client()

        
    def getTokenSize(self, input_data):
        if isinstance(input_data, list):
            text = ' '.join(str(item) for item in input_data)
        else:
            text = str(input_data)

        # tokens = re.findall(r"\w+|[^\w\s]", text, re.UNICODE)
        # return len(tokens)
        avg_token_length = 4  # Average number of characters per token (including spaces)
        return len(text) // avg_token_length

    # def getChunks(self, content, chunk_size_key):

        
    #     chunk_size = CHUNK_SIZES[chunk_size_key][0]
    #     overlap_size = CHUNK_SIZES[chunk_size_key][1]
    #     chunks = []
        
    #     if chunk_size_key == "all":
    #         return [content]  # Single chunk with the full content
        

    #     tokens = re.findall(r"\w+|[^\w\s]", content, re.UNICODE)
    #     current_chunk = []
    #     current_size = 0


    #     for token in tokens:
    #         current_chunk.append(token)
    #         current_size += 1
            
    #         if current_size >= chunk_size:
    #             chunks.append(" ".join(current_chunk))
    #             current_chunk = current_chunk[-overlap_size:]
    #             current_size = len(current_chunk)

    #     if current_chunk:
    #         chunks.append(" ".join(current_chunk))

    #     return chunks

    def getChunks(
        self,
        content: str,
        chunk_size_key: Optional[str] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> List[str]:
        """
        Chunk content by absolute character size (chunk_size/overlap) or by preset key (chunk_size_key).
        When chunk_size (int) is given: use character-based sliding window; overlap defaults from get_overlap_for_chunk_size.
        """
        if chunk_size is not None and chunk_size > 0:
            # Absolute character-based chunking (preferred for API)
            overlap_val = overlap if overlap is not None else get_overlap_for_chunk_size(chunk_size)
            return self._chunk_by_chars(content, chunk_size, overlap_val)
        key = chunk_size_key or "a4"
        if key == "all":
            return [content] if content else []
        chunk_tokens = CHUNK_SIZES.get(key, CHUNK_SIZES["a4"])[0]
        overlap_tokens = CHUNK_SIZES.get(key, CHUNK_SIZES["a4"])[1]
        tokens = re.findall(r"\w+|[^\w\s]", content, re.UNICODE)
        total_tokens = len(tokens)
        chunks = []
        start = 0
        while start < total_tokens:
            end = start + chunk_tokens
            chunk = tokens[start:end]
            chunks.append(" ".join(chunk))
            start += chunk_tokens - overlap_tokens
        return chunks

    def _chunk_by_chars(self, content: str, chunk_size: int, overlap: int) -> List[str]:
        """Slide window by characters; chunk_size and overlap in characters."""
        if not content or chunk_size <= 0:
            return []
        overlap = min(max(0, overlap), chunk_size - 1)
        step = chunk_size - overlap
        chunks = []
        start = 0
        while start < len(content):
            end = start + chunk_size
            chunk = content[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += step
        return chunks

    def getChunks__(self, content, chunk_size_key):
        chunk_tokens = CHUNK_SIZES[chunk_size_key][0]
        overlap_tokens = CHUNK_SIZES[chunk_size_key][1]
        
        if chunk_size_key == "all":
            return [content]  # Return whole content as a single chunk

        tokens = re.findall(r"\w+|[^\w\s]", content, re.UNICODE)
        total_tokens = len(tokens)
        
        chunks = []
        start = 0

        while start < total_tokens:
            end = start + chunk_tokens
            chunk = tokens[start:end]
            chunk_text = " ".join(chunk)
            chunks.append(chunk_text)
            
            start += chunk_tokens - overlap_tokens  # Slide window with overlap

        return chunks
    

    

    @contextmanager
    def get_client(self):
        try:
            if self.client is None:
                self.client = chromadb.PersistentClient(
                    path=self.DB_PATH,
                    settings=Settings(
                        allow_reset=True,
                        is_persistent=True
                    ),
                    tenant=DEFAULT_TENANT,
                    database=DEFAULT_DATABASE,
                )
            yield self.client
        finally:
            if self.client:
                self.client.clear_system_cache()
                self.client = None

    def close(self):
        pass
        # if self.executor:
        #     self.executor.shutdown(wait=True)
        # self.close_all_connections()
    def close_all_connections(self):
        pass

    def __del__(self):
        self.close()


    # def _create_chunks(self,content, chunk_size_key):

    #     chunk_size = CHUNK_SIZES[chunk_size_key][0]*2
    #     overlap_size = CHUNK_SIZES[chunk_size_key][1]*2
    #     # Generate chunks with overlap
    #     chunks = [
    #         content[i:i + chunk_size]
    #         for i in range(0, len(content) - chunk_size + 1, chunk_size - overlap_size)
    #     ]
    #     return chunks

    def _prepare_chunk_metadata(
        self,
        file_name: str,
        chunk_index: int,
        chunk: str,
        total_chunks: int,
        base_metadata: Dict[str, Any],
        doc_level_meta: Dict[str, Any],
        extract_nlp_metadata: bool,
    ) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """
        Prepare (doc_id, document, meta) for one chunk. Safe to call from threads.
        Returns None on failure so caller can skip that chunk.
        """
        try:
            meta: Dict[str, Any] = {
                "filename": file_name,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
            }
            meta.update(base_metadata)
            if doc_level_meta:
                meta.update(doc_level_meta)

            if extract_nlp_metadata and chunk:
                classifier = self._get_classifier()
                if classifier:
                    has_doc_level = "nlp_subject" in meta
                    if not has_doc_level:
                        chunk_nlp = self.extract_content_metadata(chunk)
                        meta.update(chunk_nlp)
                    else:
                        kw = classifier.extract_keywords(chunk, top_n=10)
                        if kw:
                            meta["nlp_keywords"] = ", ".join(str(k) for k in kw[:10])
                        summary = classifier.summarize(chunk, max_length=200)
                        if summary:
                            meta["nlp_summary"] = summary[:500]
                        facts = classifier.extract_key_facts(chunk)
                        if facts:
                            meta["nlp_key_facts"] = " | ".join(str(f) for f in facts[:5])

            for k, v in list(meta.items()):
                if v is not None and not isinstance(v, (str, int, float, bool)):
                    meta[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

            doc_id = f"{file_name}_chunk_{chunk_index}"
            return (doc_id, chunk, meta)
        except Exception as e:
            logger.debug("NLP prepare failed for chunk %s of %s: %s", chunk_index, file_name, e)
            return None

    def addFileToRag(
        self,
        file_name: str,
        content: str,
        chunk_size_key: str = "a4",
        metadata: Optional[Dict] = None,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        extract_nlp_metadata: bool = True,
    ) -> bool:
        """
        Add file to RAG with optional NLP metadata extraction.

        Uses a thread pool to prepare per-chunk NLP metadata in parallel, then
        adds documents to Chroma in batches for faster embedding and indexing.
        """
        logger.info("Adding file to RAG: %s, chunk_size_key: %s, chunk_size: %s", file_name, chunk_size_key, chunk_size)
        if not self._ensure_initialized():
            logger.error("RAG manager not initialized")
            return False
        if not self.client or not self.collection:
            try:
                self._initialize_client()
            except Exception as e:
                logger.error("Failed to initialize client: %s", e)
                return False

        content = str(content)
        if len(content) < 10:
            return False

        try:
            logger.info("Processing file: %s, content length: %s", file_name, len(content))

            if chunk_size is not None and chunk_size >= 1000:
                chunks = self.getChunks(content, chunk_size_key=None, chunk_size=chunk_size, overlap=overlap)
            else:
                chunks = self.getChunks(content, chunk_size_key=chunk_size_key)
            total_chunks = len(chunks)
            logger.info("Created %s chunks (parallel NLP + batch add)", total_chunks)

            doc_level_meta: Dict[str, Any] = {}
            if extract_nlp_metadata:
                sample = content[:10_000]
                doc_level_meta = self.extract_content_metadata(sample)
                doc_level_meta.pop("nlp_summary", None)
                doc_level_meta.pop("nlp_key_facts", None)
                doc_level_meta.pop("nlp_keywords", None)
                logger.info(
                    "Document-level NLP metadata: subject=%s, type=%s, lang=%s",
                    doc_level_meta.get("nlp_subject"),
                    doc_level_meta.get("nlp_document_type"),
                    doc_level_meta.get("nlp_language"),
                )

            base_meta = dict(metadata) if metadata else {}
            base_meta["filename"] = file_name

            # Prepare all chunk metadata in parallel (thread pool)
            futures = [
                _nlp_prepare_pool.submit(
                    self._prepare_chunk_metadata,
                    file_name,
                    i,
                    chunk,
                    total_chunks,
                    base_meta.copy(),
                    doc_level_meta,
                    extract_nlp_metadata,
                )
                for i, chunk in enumerate(chunks)
            ]
            prepared: List[Tuple[str, str, Dict[str, Any]]] = []
            for i, fut in enumerate(concurrent.futures.as_completed(futures)):
                result = fut.result()
                if result is not None:
                    prepared.append(result)
                else:
                    logger.warning("Skipped chunk %s of file %s (prepare failed)", i, file_name)

            # Sort by chunk_index so batch order is stable
            prepared.sort(key=lambda x: x[2].get("chunk_index", 0))

            # Batch add to Chroma
            batch_size = RAG_ADD_BATCH_SIZE
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                ids_batch = [p[0] for p in batch]
                docs_batch = [p[1] for p in batch]
                metas_batch = [p[2] for p in batch]
                self.collection.add(ids=ids_batch, documents=docs_batch, metadatas=metas_batch)
                if total_chunks > batch_size:
                    logger.info("RAG progress: %s added batch %s-%s/%s", file_name, start + 1, start + len(batch), total_chunks)

            logger.info("Successfully processed file %s (%s chunks)", file_name, len(prepared))
            return len(prepared) > 0

        except Exception as e:
            logger.error("Exception in addFileToRag for file %s: %s", file_name, e)
            return False
        
    def adDocumentToRag(
        self,
        file_name: str,
        document: str,
        metadata: Optional[Dict] = None,
        extract_nlp_metadata: bool = True,
    ) -> bool:
        """Add document to RAG (legacy name). Use addDocumentToRag for API."""
        return self.addDocumentToRag(file_name, document, metadata, extract_nlp_metadata=extract_nlp_metadata)

    def addDocumentToRag(
        self,
        file_name: str,
        document: str,
        metadata: Optional[Dict] = None,
        extract_nlp_metadata: bool = True,
    ) -> bool:
        """
        Add a single document (chunk) to RAG with optional NLP metadata.

        When *extract_nlp_metadata* is True, the content classifier extracts
        per-chunk keywords, summary, and key facts.  Document-level metadata
        (subject, language, etc.) is expected to already be present in
        *metadata* when called from addFileToRag; if absent it will be
        extracted here as well.
        """
        if not self._ensure_initialized():
            logger.error("RAG manager not initialized")
            return False
        try:
            meta: Dict[str, Any] = {"filename": file_name}
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        meta[k] = v
                    else:
                        meta[k] = json.dumps(v) if isinstance(v, (dict, list)) else str(v)

            if extract_nlp_metadata and document:
                try:
                    classifier = self._get_classifier()
                    if classifier:
                        has_doc_level = "nlp_subject" in meta
                        if not has_doc_level:
                            chunk_nlp = self.extract_content_metadata(document)
                            meta.update(chunk_nlp)
                        else:
                            kw = classifier.extract_keywords(document, top_n=10)
                            if kw:
                                meta["nlp_keywords"] = ", ".join(str(k) for k in kw[:10])
                            summary = classifier.summarize(document, max_length=200)
                            if summary:
                                meta["nlp_summary"] = summary[:500]
                            facts = classifier.extract_key_facts(document)
                            if facts:
                                meta["nlp_key_facts"] = " | ".join(str(f) for f in facts[:5])
                except Exception as e:
                    logger.debug(f"NLP metadata extraction skipped for chunk: {e}")

            doc_id = str(meta.get("document_id", f"{file_name}_{int(time.time() * 1000000)}"))
            self.collection.add(documents=[document], ids=[doc_id], metadatas=[meta])
            return True
        except Exception as e:
            logger.exception("addDocumentToRag failed: %s", e)
            return False

    def addDocumentsToRag(
        self,
        file_id: str,
        documents_list: List[Dict[str, Any]],
        extract_nlp_metadata: bool = True,
    ) -> bool:
        """
        Batch add documents to RAG with optional NLP metadata.

        Each item must have "id" and "content".
        When *extract_nlp_metadata* is True, each document's content is
        classified and the flat metadata is stored in ChromaDB alongside
        the vector embedding.
        """
        if not documents_list:
            return True
        if not self._ensure_initialized():
            logger.error("RAG manager not initialized")
            return False
        try:
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for doc in documents_list:
                doc_id = str(doc["id"])
                content = doc["content"]
                meta: Dict[str, Any] = {
                    "document_id": doc_id,
                    "filename": str(file_id),
                }

                if extract_nlp_metadata and content:
                    try:
                        nlp_meta = self.extract_content_metadata(content)
                        meta.update(nlp_meta)
                    except Exception as e:
                        logger.debug(f"NLP metadata skipped for doc {doc_id}: {e}")

                ids.append(doc_id)
                documents.append(content)
                metadatas.append(meta)

            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            return True
        except Exception as e:
            logger.exception("addDocumentsToRag failed: %s", e)
            return False

    # Verb forms that match when reranking (e.g. "who saw the dragon" -> "Malfoy had seen the dragon")
    _verb_equivalents = {
        "saw": ["seen", "see", "saw"],
        "seen": ["saw", "see", "seen"],
        "see": ["saw", "seen", "see"],
        "got": ["gotten", "get", "got"],
        "gotten": ["got", "get", "gotten"],
        "get": ["got", "gotten", "get"],
        "went": ["gone", "go", "went"],
        "gone": ["went", "go", "gone"],
        "go": ["went", "gone", "go"],
        "took": ["taken", "take", "took"],
        "taken": ["took", "take", "taken"],
        "take": ["took", "taken", "take"],
        "knew": ["known", "know", "knew"],
        "known": ["knew", "know", "known"],
        "know": ["knew", "known", "know"],
        "said": ["say", "said"],
        "say": ["said", "say"],
        "had": ["have", "has", "had"],
        "have": ["had", "has", "have"],
        "has": ["had", "have", "has"],
    }

    def _answer_phrase_variants(self, query_str: str) -> List[str]:
        """E.g. 'who saw the dragon' -> ['saw the dragon', 'seen the dragon', 'see the dragon']."""
        q = (query_str or "").strip().lower()
        leading = r"^(who|what|where|when|why|how|which|whose)\s+(?:did\s+|does\s+|do\s+|has\s+|have\s+|had\s+)?"
        rest = re.sub(leading, "", q, flags=re.IGNORECASE).strip()
        rest = re.sub(r"[^\w\s]", " ", rest)
        rest = " ".join(rest.split())
        if len(rest) < 3:
            return []
        words = rest.split()
        if not words:
            return []
        first = words[0]
        variants = self._verb_equivalents.get(first, [first])
        phrase_tail = " ".join(words[1:])
        return [f"{v} {phrase_tail}".strip() for v in variants if v != first] + [rest]

    def _rerank_by_query_words(self, items: List[Dict], query_str: str, top_k: int = 10) -> List[Dict]:
        """Rerank: answer-phrase match (e.g. 'seen the dragon') > exact phrase > word overlap (verb equivalents) > vector score."""
        if not query_str or not items:
            return items
        query_words = set(re.findall(r"\w+", query_str.lower()))
        if not query_words:
            return items

        def _normalize_phrase(text: str) -> str:
            t = (text or "").lower()
            t = re.sub(r"[^\w\s]", " ", t)
            return " ".join(t.split())

        query_phrase = _normalize_phrase(query_str)
        answer_phrases = [_normalize_phrase(p) for p in self._answer_phrase_variants(query_str)]

        def _content_matches_query_word(content: str, w: str) -> bool:
            if w in content:
                return True
            for alt in self._verb_equivalents.get(w, []):
                if alt in content:
                    return True
            return False

        def score_item(item: Dict) -> tuple:
            content = (item.get("content") or "").lower()
            if not content:
                return (0, 0, 0, 0)
            content_phrase = _normalize_phrase(content)
            answer_phrase_match = 1 if any(ap in content_phrase for ap in answer_phrases) else 0
            phrase_match = 1 if query_phrase in content_phrase else 0
            word_hits = sum(1 for w in query_words if _content_matches_query_word(content, w))
            vec_score = item.get("score") or 0
            return (answer_phrase_match, phrase_match, word_hits, vec_score)

        scored = [(score_item(item), item) for item in items]
        scored.sort(key=lambda x: (-x[0][0], -x[0][1], -x[0][2], -x[0][3]))
        return [item for _, item in scored[:top_k]]

    _MAX_MULTI_QUERIES = 5

    def _parse_multi_query(self, query) -> List[str]:
        """Parse query into up to 5 non-empty query strings (comma-separated or list)."""
        if isinstance(query, str):
            parts = [p.strip() for p in query.split(",") if p.strip()]
        else:
            parts = [str(q).strip() for q in (query if isinstance(query, (list, tuple)) else [query]) if str(q).strip()]
        return parts[: self._MAX_MULTI_QUERIES]

    def queryCollection(
        self,
        query,
        keywords=[],
        limit=10,
        debug=False,
        min_relevance_threshold=0.0,
        subject: Optional[str] = None,
        document_type: Optional[str] = None,
        language: Optional[str] = None,
        sentiment: Optional[str] = None,
        reading_level: Optional[str] = None,
        min_word_count: Optional[int] = None,
        keyword_contains: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
        show_meta: bool = True,
    ):
        """
        Vector search over ChromaDB with optional NLP metadata filtering,
        then rerank by lexical match.

        Metadata filters (all optional) are applied at the ChromaDB level,
        narrowing the search *before* vector scoring:
          - subject:          e.g. "Technology", "Finance"
          - document_type:    e.g. "question_answer", "tutorial"
          - language:         e.g. "en", "zh"
          - sentiment:        "positive", "negative", "neutral"
          - reading_level:    "elementary", "intermediate", "advanced", "academic"
          - min_word_count:   minimum word count per chunk
          - keyword_contains: substring match within nlp_keywords field
          - metadata_filter:  raw ChromaDB where dict (merged with above)
          - show_meta:        when True (default), each result includes ``meta_data``
                              (parsed rich NLP JSON) and the flat ``metadata`` dict;
                              when False only ``doc``, ``content``, ``filename``,
                              and ``score`` are returned.

        Supports up to 5 comma-separated queries; runs one batch vector
        query and merges results by best score per document, then reranks.
        """
        queries = self._parse_multi_query(query)
        if not queries:
            return []
        query_str = " ".join(queries)
        logger.info(f"QUERY COLLECTION: {query_str} (queries={len(queries)})")
        if not self._ensure_initialized():
            logger.error("Failed to initialize RAG manager")
            return []

        if isinstance(keywords, list):
            extra = keywords
        else:
            extra = [keywords] if keywords else []
        query_texts = list(queries) + extra

        where_clause = self._build_metadata_where_clause(
            subject=subject,
            document_type=document_type,
            language=language,
            sentiment=sentiment,
            reading_level=reading_level,
            min_word_count=min_word_count,
            keyword_contains=keyword_contains,
            extra_where=metadata_filter,
        )
        if where_clause and debug:
            logger.info(f"Metadata where clause: {where_clause}")

        try:
            if self.collection.count() == 0:
                logger.info("Collection is empty, no documents to search")
                return []
            coll_count = self.collection.count()
            per_query_n = max(limit * 4, 25) if len(queries) > 1 else max(limit * 12, 80)
            n_results = min(per_query_n, coll_count)

            query_kwargs: Dict[str, Any] = {
                "query_texts": query_texts,
                "n_results": n_results,
                "include": ["documents", "distances", "metadatas"],
            }
            if where_clause:
                query_kwargs["where"] = where_clause

            results = self.collection.query(**query_kwargs)
            if not results["ids"] or not results["ids"][0]:
                logger.info("No results found")
                return []
        except Exception as e:
            logger.error(f"Error in queryCollection: {e}")
            return []

        seen: Dict[str, Dict[str, Any]] = {}
        num_queries = len(results["ids"])
        for i in range(num_queries):
            ids_i = results["ids"][i] or []
            docs_i = results["documents"][i] or []
            dist_i = results["distances"][i] or []
            meta_i = results["metadatas"][i] or []
            for j, doc_id in enumerate(ids_i):
                if j >= len(docs_i) or j >= len(dist_i):
                    continue
                score = 1.0 - dist_i[j]
                meta = dict(meta_i[j]) if j < len(meta_i) and isinstance(meta_i[j], dict) else {}
                if not meta.get("document_id"):
                    meta["document_id"] = doc_id
                content = docs_i[j] if isinstance(docs_i[j], str) else (docs_i[j] or "")

                # Parse rich NLP metadata from the stored JSON string
                meta_data: Dict[str, Any] = {}
                if show_meta:
                    raw_meta_json = meta.get("nlp_meta_json", "")
                    if raw_meta_json:
                        try:
                            meta_data = json.loads(raw_meta_json)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            meta_data = {}

                if doc_id not in seen or score > seen[doc_id]["score"]:
                    seen[doc_id] = {
                        # Primary document content — both keys for forward + backward compat
                        "doc": content,
                        "content": content,
                        "filename": meta.get("filename", "unknown"),
                        # Always include document_id for downstream DB lookup regardless of show_meta
                        "document_id": meta.get("document_id") or doc_id,
                        # Flat ChromaDB metadata (for backward compat, omitted when show_meta=False)
                        "metadata": meta if show_meta else {},
                        # Rich structured NLP metadata parsed from nlp_meta_json
                        "meta_data": meta_data,
                        "score": score,
                    }

        all_results = list(seen.values())
        all_results.sort(key=lambda x: -x["score"])

        all_results = self._rerank_by_query_words(all_results, query_str, top_k=limit * 3)

        final = []
        for item in all_results:
            if (item.get("score") or 0) < min_relevance_threshold:
                continue
            final.append(item)
            if len(final) >= limit:
                break
        return final

    def queryCollection__(self, query: str, keywords: List[str] = None, limit: int = 10, 
                       debug: bool = False, min_relevance_threshold: float = 0.6) -> List[Dict]:
        """
        Perform search over the RAG collection using ChromaDB - simplified working version.
        Returns results in the same format as Milvus version.
        """
        print(f"\n--- QUERY COLLECTION START: {query} ---\n")
        
        # Simple initialization check - create client if needed
        if not self.client or not self.collection:
            try:
                self._initialize_client()
            except Exception as e:
                logger.error(f"Failed to initialize client for query: {e}")
                return []

        # Prepare query - keep it simple like the working version
        keywords = keywords or []
        
        try:
            # Check collection count first
            collection_count = self.collection.count()
            
            if collection_count == 0:
                print("Collection is empty, no documents to search")
                return []
            

            if isinstance(query, str):
                query_list = query.split(" ")
            else:
                query_list = query

            query_list = query_list + keywords if isinstance(keywords, list) else query_list

            # Use simple approach like the working version
            n_results = min(limit * 3, 50)  # Get more results for filtering
            results = self.collection.query(
                query_texts=query_list,  # Simple single query like working version
                n_results=n_results,
                include=["documents", "distances", "metadatas"],
            )
            

            print(f"Results: {results}")
            if not results["ids"] or not results["ids"][0]:
                print("No results found")
                return []

            # Process results - simplified like working version
            combined = list(zip(results["ids"][0], results["documents"][0], results["distances"][0], results["metadatas"][0]))
            combined.sort(key=lambda x: x[2])  # Sort by distance (ascending)

            # Format results to match working version
            final_res = []
            if combined:
                first_doc_score = max(0, 1.0 - combined[0][2])  # Convert distance to similarity
                cutoff_score = first_doc_score * 0.8
                
                for i, (doc_id, doc_content, distance, metadata) in enumerate(combined):
                    try:
                        # Parse document content
                        doc_json = json.loads(doc_content)
                        content = doc_json.get("content", "")
                        
                        if not content:
                            continue
                        
                        # Convert distance to similarity score
                        similarity_score = distance
                        
                        # Apply threshold filtering like working version
                        if similarity_score >= min_relevance_threshold:
                            if debug:
                                print(f"Filtered out result with score {similarity_score:.3f} < {cutoff_score:.3f}")
                            break
                            
                        # Reconstruct metadata from flattened format
                        reconstructed_metadata = {}
                        for key, value in metadata.items():
                            if key == "filename":
                                continue  # Skip filename as it's handled separately
                            try:
                                # Try to parse JSON strings back to objects
                                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                                    reconstructed_metadata[key] = json.loads(value)
                                else:
                                    reconstructed_metadata[key] = value
                            except:
                                # If parsing fails, keep as string
                                reconstructed_metadata[key] = value
                        final_res.append({
                            "content": content,
                            "filename": metadata.get("filename", "unknown"),
                            "metadata": reconstructed_metadata,
                            "score": similarity_score
                        })
                        
                        if len(final_res) >= limit:
                            break
                                
                    except Exception as e:
                        print(f"Error processing result: {e}")
                        continue

            print(f"\n--- QUERY COLLECTION END: Found {len(final_res)} results (min relevance: {min_relevance_threshold}) ---\n")
            return final_res

        except Exception as e:
            print(f"Error in queryCollection: {e}")
            import traceback
            traceback.print_exc()
            return []



   


    def deleteRag(self) -> bool:
        """Delete all RAG data with improved error handling."""
        logger.info(f"Deleting RAG data for {self.path}")
        
        try:
            with self._lock:
                success = True
                
                # Delete collections
                if self.client:
                    try:
                        self.client.delete_collection(self.collection_name)
                        logger.info(f"Deleted collection: {self.collection_name}")
                    except Exception as e:
                                logger.error(f"Error deleting collection {self.collection_name}: {e}")
                                success = False

                try:
                    self.client.delete_collection(self.image_collection_name)
                    logger.info(f"Deleted image collection: {self.image_collection_name}")
                except Exception as e:
                            logger.error(f"Error deleting image collection {self.image_collection_name}: {e}")
                            success = False

            try:
                self.client.reset()
                logger.info("Client reset completed")
            except Exception as e:
                logger.error(f"Error resetting client: {e}")
                success = False
                
                # Force cleanup
                self._force_cleanup()
                
                return success
                
        except Exception as e:
            logger.error(f"Error deleting RAG: {e}")
            return False

        

    def deleteFileFromRag(self, file_name: str) -> bool:
        """Delete specific file from RAG with improved error handling."""
        logger.info(f"Deleting file from RAG: {file_name}")
        
        # Ensure initialization
        if not self._ensure_initialized():
            logger.error("RAG manager not initialized")
            return False
        
        try:
            if not self.collection:
                logger.error("Collection not available")
                return False
            
            # Get count before deletion
            docs_before = self.collection.count()
            logger.debug(f"Documents before deletion: {docs_before}")
            
            # Delete documents with matching filename in metadata
            # Support both exact filename and chunk-based filenames
            self.collection.delete(where={"filename": file_name})
            
            # Also delete chunk-based files
            try:
                # Query for chunk-based files that start with the filename
                chunk_results = self.collection.get(
                    where={"filename": {"$regex": f"^{re.escape(file_name)}_chunk_"}},
                    include=["metadatas"]
                )
                
                if chunk_results and chunk_results.get("metadatas"):
                    chunk_filenames = [meta.get("filename") for meta in chunk_results["metadatas"] if meta.get("filename")]
                    for chunk_filename in chunk_filenames:
                        if chunk_filename != file_name:  # Avoid double deletion
                            self.collection.delete(where={"filename": chunk_filename})
                            
            except Exception as e:
                logger.warning(f"Could not delete chunk files for {file_name}: {e}")
            
            # Get count after deletion
            docs_after = self.collection.count()
            deleted_count = docs_before - docs_after
            
            logger.info(f"Deleted {deleted_count} documents for file {file_name}")
            logger.debug(f"Documents after deletion: {docs_after}")
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting file {file_name}: {e}")
            return False

    def deleteDocumentFromRag(self, document_id: str) -> bool:
        """Remove a single document's vector by document_id (API)."""
        if not self._ensure_initialized():
            return False
        try:
            if not self.collection:
                return False
            doc_id_str = str(document_id)
            self.collection.delete(where={"document_id": doc_id_str})
            return True
        except Exception as e:
            logger.warning("deleteDocumentFromRag %s: %s", document_id, e)
            return False

    def query_collection_results(
        self,
        query: str,
        limit: int = 10,
        keywords: Optional[List[str]] = None,
        min_relevance_threshold: float = 0.0,
        subject: Optional[str] = None,
        document_type: Optional[str] = None,
        language: Optional[str] = None,
        sentiment: Optional[str] = None,
        reading_level: Optional[str] = None,
        min_word_count: Optional[int] = None,
        keyword_contains: Optional[str] = None,
        metadata_filter: Optional[Dict] = None,
        show_meta: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Query RAG and return structured results.

        Each result follows the canonical document storage format::

            {
                "doc":          str,   # the document corpus text
                "content":      str,   # alias for doc (backward compat)
                "filename":     str,
                "score":        float,
                # only present when show_meta=True:
                "meta_data":    dict,  # full rich NLP metadata (parsed nlp_meta_json)
                "metadata":     dict,  # flat ChromaDB metadata fields
                "nlp_metadata": dict,  # nlp_* fields extracted from flat metadata
            }

        All NLP metadata filter params are applied at the ChromaDB index level
        (via *where* clause) before vector scoring.

        Args:
            show_meta: When True (default), include ``meta_data``, ``metadata``,
                       and ``nlp_metadata`` in each result.  When False, only
                       ``doc``, ``content``, ``filename``, and ``score`` are returned,
                       keeping the response lean for downstream consumers that only
                       need the document text.
        """
        kw = list(keywords) if keywords else []
        raw = self.queryCollection(
            query=query,
            keywords=kw,
            limit=limit,
            min_relevance_threshold=min_relevance_threshold,
            subject=subject,
            document_type=document_type,
            language=language,
            sentiment=sentiment,
            reading_level=reading_level,
            min_word_count=min_word_count,
            keyword_contains=keyword_contains,
            metadata_filter=metadata_filter,
            show_meta=show_meta,
        )
        out: List[Dict[str, Any]] = []
        for r in raw:
            content = r.get("doc") or r.get("content") or ""

            if not show_meta:
                out.append({
                    "doc": content,
                    "content": content,
                    "filename": r.get("filename") or "",
                    "document_id": r.get("document_id") or "",
                    "score": float(r.get("score") or 0),
                })
                continue

            # Resolve flat metadata dict
            meta = r.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta) if meta else {}
                except (json.JSONDecodeError, TypeError):
                    meta = {}
            if not isinstance(meta, dict):
                meta = {}

            doc_id = meta.get("document_id") or r.get("filename") or ""
            meta["document_id"] = doc_id

            # Rich structured metadata (may already be parsed by queryCollection)
            meta_data = r.get("meta_data") or {}
            if not isinstance(meta_data, dict):
                meta_data = {}

            # Flat nlp_* fields for backward-compat consumers
            nlp_meta = {k: v for k, v in meta.items() if k.startswith("nlp_") and k != "nlp_meta_json"}

            out.append({
                "doc": content,
                "content": content,
                "filename": r.get("filename") or "",
                "metadata": meta,
                "meta_data": meta_data,
                "nlp_metadata": nlp_meta,
                "score": float(r.get("score") or 0),
            })
        return out


# Global cleanup function
def cleanup_all_rag_managers():
    """Clean up all RAG managers and global resources."""
    try:
        _rag_registry.cleanup_all()
        _nlp_prepare_pool.shutdown(wait=True)
        _thread_pool.shutdown(wait=True)
        logger.info("All RAG managers cleaned up")
    except Exception as e:
        logger.error(f"Error during global cleanup: {e}")

# Register cleanup on exit
atexit.register(cleanup_all_rag_managers)

# Factory function for easy access
def get_rag_manager(path: str) -> RAGManager:
    """
    Factory function to get a RAG manager instance.
    This is the recommended way to get RAG manager instances.
    
    Args:
        path: Path to the RAG data directory
        
    Returns:
        RAGManager instance
    """
    return RAGManager.get_instance(path)

# Health check function
def health_check_all_managers() -> Dict[str, Any]:
    """
    Perform health check on all active RAG managers and global components.
    
    Returns:
        Dict containing health status of all components
    """
    try:
        health_status = {
            "global_embedding_manager": _global_embedding_manager.health_check(),
            "active_managers": len(_rag_registry._managers),
            "thread_pool_active": not _thread_pool._shutdown,
            "managers": {}
        }
        
        # Check individual managers
        with _rag_registry._lock:
            for path, manager in _rag_registry._managers.items():
                try:
                    health_status["managers"][path] = manager._health_check()
                except Exception as e:
                    health_status["managers"][path] = {"error": str(e)}
        
        return health_status
        
    except Exception as e:
        return {"error": f"Health check failed: {e}"}


# Async factory function
async def get_rag_manager_async(path: str) -> RAGManager:
    """
    Async factory function to get a RAG manager instance with async initialization.
    
    Args:
        path: Path to the RAG data directory
        
    Returns:
        RAGManager instance
    """
    loop = asyncio.get_event_loop()
    
    def _get_manager():
        manager = RAGManager.get_instance(path)
        manager._ensure_initialized()  # Ensure it's initialized
        return manager
    
    return await loop.run_in_executor(_thread_pool, _get_manager)


if __name__ == "__main__":
    basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "")
    rag_client = get_rag_manager(os.path.join(basepath, "db", "test"))

    filepath = os.path.join(basepath, "dummydata", "AI_media.pdf")
    try:
        from core.knowledgebase.filemanager import FileManager
    except ImportError:
        from knowledgebase.filemanager import FileManager
    file_manager = FileManager(custom_base_path=basepath)
    content = file_manager.get_file_content_from_path(filepath)
    rag_client.addFileToRag("AI_media.pdf", content, chunk_size=2000,overlap=500)

    while True:
        query = input("Enter a query: ")
        documents = rag_client.queryCollection(query=query, limit=4)
        with open("documents.json", "w") as f:
            json.dump(documents, f)
        for document in documents:
            print("document", document["content"], "\n--------------------------------\n")




import json
import hashlib
import math
from typing import Dict, Any, List, Optional, Tuple, Set
import re
import os
import sys
import threading
import unicodedata
import time
import logging
import atexit
import gc
from collections import Counter
from functools import lru_cache

# -------------------------------
# PROJECT ROOT SETUP
# -------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

CACHE_PATH = "/var/www/intgr8-v5/tmp_cache"
IS_PRODUCTION = os.path.exists(CACHE_PATH)
IS_DEVELOPMENT = not IS_PRODUCTION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------
# LAZY IMPORTS
# -------------------------------
_spacy = None
_sentence_transformers = None
_keybert = None
_sumy = None


def _lazy_import_spacy():
    global _spacy
    if _spacy is None:
        try:
            import spacy
            _spacy = spacy
        except ImportError as e:
            logger.warning(f"Could not import spacy: {e}")
            _spacy = False
    return _spacy if _spacy is not False else None


def _lazy_import_sentence_transformers():
    global _sentence_transformers
    if _sentence_transformers is None:
        try:
            from sentence_transformers import SentenceTransformer, util
            _sentence_transformers = {'SentenceTransformer': SentenceTransformer, 'util': util}
        except ImportError as e:
            logger.warning(f"Could not import sentence_transformers: {e}")
            _sentence_transformers = False
    return _sentence_transformers if _sentence_transformers is not False else None


def _lazy_import_keybert():
    global _keybert
    if _keybert is None:
        try:
            from keybert import KeyBERT
            _keybert = KeyBERT
        except ImportError as e:
            logger.warning(f"Could not import keybert: {e}")
            _keybert = False
    return _keybert if _keybert is not False else None


def _lazy_import_sumy():
    global _sumy
    if _sumy is None:
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            _sumy = {
                'PlaintextParser': PlaintextParser,
                'Tokenizer': Tokenizer,
                'TextRankSummarizer': TextRankSummarizer
            }
        except ImportError as e:
            logger.warning(f"Could not import sumy: {e}")
            _sumy = False
    return _sumy if _sumy is not False else None


# -------------------------------
# HASH-BASED RESULT CACHE
# -------------------------------
class _ResultCache:
    """Thread-safe LRU cache keyed by text hash instead of full text."""

    def __init__(self, maxsize: int = 1024):
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._maxsize = maxsize
        self._lock = threading.Lock()
        self._ttl = 3600

    def _hash_key(self, text: str, prefix: str = "") -> str:
        digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]
        return f"{prefix}:{digest}" if prefix else digest

    def get(self, text: str, prefix: str = "") -> Optional[Any]:
        key = self._hash_key(text, prefix)
        with self._lock:
            entry = self._cache.get(key)
            if entry and (time.time() - entry[0]) < self._ttl:
                return entry[1]
            if entry:
                del self._cache[key]
        return None

    def set(self, text: str, value: Any, prefix: str = "") -> None:
        key = self._hash_key(text, prefix)
        with self._lock:
            if len(self._cache) >= self._maxsize:
                oldest_key = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest_key]
            self._cache[key] = (time.time(), value)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


_result_cache = _ResultCache(maxsize=2048)


# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "ave", "vs",
    "etc", "inc", "ltd", "corp", "dept", "univ", "approx", "est",
    "govt", "assn", "bros", "fig", "vol", "no", "gen", "adm",
    "sgt", "cpl", "pvt", "capt", "cmdr", "lt", "col", "maj",
    "u.s", "u.k", "u.n", "e.g", "i.e", "a.m", "p.m",
})

SUBJECT_LABELS = [
    "Legal", "Marketing", "Technology", "Finance", "Business", "Economics",
    "Political", "Government", "Health", "Medicine", "Science", "Engineering",
    "Environment", "Energy", "AI & Machine Learning", "Cybersecurity",
    "Data Science", "Sports", "Entertainment", "Education", "News",
    "Social Media", "Agriculture", "History", "Philosophy", "Psychology",
    "Logistics & Supply Chain", "Real Estate", "Human Resources",
    "Automotive", "Aerospace", "Telecommunications",
]

DOMAIN_KEYWORDS = {
    "Technology": [
        "software", "technology", "computer", "digital", "internet",
        "cloud", "algorithm", "programming", "database", "hardware",
        "api", "platform", "server", "network", "silicon",
    ],
    "AI & Machine Learning": [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "nlp", "natural language", "transformer",
        "training data", "model", "inference", "gpt", "llm",
    ],
    "Finance": [
        "money", "investment", "financial", "market", "stock", "economy",
        "banking", "revenue", "profit", "interest rate", "inflation",
        "asset", "equity", "dividend", "portfolio",
    ],
    "Health": [
        "health", "medical", "patient", "doctor", "disease", "treatment",
        "hospital", "clinical", "pharmaceutical", "vaccine", "therapy",
        "diagnosis", "symptom", "surgery",
    ],
    "Legal": [
        "law", "legal", "court", "judge", "regulation", "compliance",
        "statute", "litigation", "attorney", "contract", "jurisdiction",
        "precedent", "plaintiff", "defendant",
    ],
    "Business": [
        "business", "company", "corporate", "management", "strategy",
        "startup", "acquisition", "merger", "revenue", "stakeholder",
        "supply chain", "logistics", "operations",
    ],
    "Education": [
        "education", "school", "university", "student", "teacher",
        "curriculum", "learning", "classroom", "exam", "academic",
        "grade", "course", "lecture", "scholarship",
    ],
    "Science": [
        "research", "experiment", "hypothesis", "scientific", "physics",
        "chemistry", "biology", "laboratory", "observation", "theory",
        "molecule", "atom", "cell", "genome",
    ],
    "Environment": [
        "environment", "climate", "pollution", "sustainability",
        "renewable", "ecosystem", "biodiversity", "carbon", "emission",
        "conservation", "deforestation", "recycling",
    ],
    "Political": [
        "government", "policy", "election", "democracy", "legislation",
        "congress", "parliament", "diplomacy", "voter", "campaign",
        "political party", "referendum",
    ],
    "Sports": [
        "sports", "game", "team", "player", "championship", "tournament",
        "athlete", "coach", "score", "stadium", "league", "season",
    ],
    "History": [
        "history", "ancient", "medieval", "century", "civilization",
        "empire", "dynasty", "war", "revolution", "colonial",
    ],
    "Agriculture": [
        "agriculture", "farming", "crop", "soil", "irrigation",
        "harvest", "fertilizer", "plantation", "livestock", "cultivation",
    ],
}

DOCUMENT_TYPE_PATTERNS = {
    "question_answer": [
        r"(?:^|\n)\s*(?:Q\d*[.:\)]|question\s*\d*[.:\)]|\d+\.\s+(?:what|how|why|when|where|which|who|explain|describe|define|discuss|state|write|name|give|distinguish))",
        r"\bans(?:wer)?[\s]*[:.]",
        r"\b(?:answer|solution|explanation)\s*[:]\s*",
    ],
    "tutorial": [
        r"(?:step\s+\d+|first(?:ly)?|second(?:ly)?|third(?:ly)?|finally)[,:\s]",
        r"\b(?:how to|guide|tutorial|walkthrough|instructions)\b",
        r"\b(?:prerequisite|requirement|setup|install|configure)\b",
    ],
    "report": [
        r"\b(?:executive summary|introduction|methodology|findings|conclusion|recommendation|abstract)\b",
        r"\b(?:table of contents|references|appendix|bibliography)\b",
        r"\b(?:figure\s+\d+|table\s+\d+|chart\s+\d+)\b",
    ],
    "legal": [
        r"\b(?:whereas|hereinafter|notwithstanding|pursuant to|shall be|hereby)\b",
        r"\b(?:article\s+\d+|section\s+\d+|clause\s+\d+)\b",
        r"\b(?:plaintiff|defendant|jurisdiction|court|tribunal)\b",
    ],
    "narrative": [
        r'\b(?:once upon|long ago|in the beginning)\b',
        r'\b(?:he said|she said|they said|replied|whispered|shouted)\b',
        r'\b(?:chapter\s+\d+|part\s+\d+)\b',
    ],
    "technical": [
        r"\b(?:api|endpoint|function|class|method|parameter|return|module)\b",
        r"```[\w]*\n",
        r"\b(?:implementation|architecture|interface|protocol|specification)\b",
    ],
    "news": [
        r"\b(?:reported|according to|sources say|officials said|press release)\b",
        r"\b(?:breaking|update|exclusive|developing|confirmed)\b",
        r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday),?\s+\w+\s+\d{1,2}\b",
    ],
    "academic": [
        r"\b(?:abstract|introduction|literature review|methodology|discussion|conclusion)\b",
        r"\b(?:hypothesis|variables|sample size|statistical|p-value|correlation)\b",
        r"\b(?:et al\.|ibid\.|doi:|issn:|journal of)\b",
    ],
    "conversational": [
        r"\b(?:hi|hello|hey|thanks|thank you|bye|goodbye|please|sorry)\b",
        r"\b(?:i think|i feel|in my opinion|personally|honestly|actually)\b",
        r"(?:^|\n)\s*(?:me:|you:|user:|assistant:|bot:)",
    ],
    "product": [
        r"\b(?:price|buy|purchase|order|shipping|discount|offer|sale|in stock)\b",
        r"\b(?:features|specifications|dimensions|weight|material|color|size)\b",
        r"\b(?:warranty|return policy|customer review|rating|stars)\b",
    ],
}

# ─────────────────────────────────────────────
# TEMPORAL PATTERNS
# ─────────────────────────────────────────────

TEMPORAL_PATTERNS: List[str] = [
    r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b',
    r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|'
    r'Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
    r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|'
    r'Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
    r'\s+\d{4}\b',
    r'\b(?:Q[1-4]|quarter\s+[1-4])\s+\d{4}\b',
    r'\bFY\s*\d{2,4}\b',
    r'\b(?:19|20)\d{2}\b',
    r'\b(?:yesterday|today|tomorrow)\b',
    r'\b(?:last|next|this)\s+(?:week|month|year|quarter|decade|century)\b',
    r'\b\d+\s+(?:years?|months?|weeks?|days?|hours?)\s+(?:ago|later|before|after|earlier)\b',
    r'\b(?:early|mid|late)\s+(?:19|20)\d{2}s?\b',
    r'\b(?:since|until|from|between|during)\s+\d{4}\b',
]

STOP_WORDS = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "this", "that", "with", "have", "from", "they", "been",
    "more", "when", "will", "some", "than", "them", "each", "make",
    "like", "long", "look", "many", "then", "what", "were", "into",
    "time", "very", "just", "know", "take", "come", "could", "good",
    "most", "also", "back", "over", "such", "only", "other", "their",
    "which", "about", "would", "there", "these", "after", "think",
    "being", "because", "does", "should", "much", "still", "well",
    "before", "here", "through", "where", "between", "both", "under",
    "never", "same", "another", "while",
})

SENTIMENT_POSITIVE = frozenset({
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "positive", "success", "benefit", "improve", "growth", "gain",
    "achievement", "advantage", "progress", "innovative", "efficient",
    "effective", "profitable", "outstanding", "remarkable", "ideal",
    "optimal", "superior", "thriving", "promising", "breakthrough",
})

SENTIMENT_NEGATIVE = frozenset({
    "bad", "poor", "terrible", "awful", "negative", "failure", "loss",
    "decline", "decrease", "risk", "threat", "damage", "harm", "crisis",
    "problem", "issue", "concern", "danger", "deficit", "bankruptcy",
    "collapse", "recession", "downturn", "catastrophe", "devastating",
    "controversial", "corrupt", "violation", "penalty",
})


# -------------------------------
# GLOBAL MODEL MANAGER
# -------------------------------
class GlobalContentClassifierManager:
    """Thread-safe singleton for managing ML models with health tracking."""

    _instance = None
    _lock = threading.RLock()
    _models: Dict[str, Any] = {
        'nlp': None,
        'embed_model': None,
        'kw_model': None,
        'subject_embeddings': None,
    }
    _initialized = False
    _init_time: Optional[float] = None
    _model_load_errors: List[str] = []

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    if IS_PRODUCTION:
                        logger.info("PRODUCTION: Loading models at startup...")
                        self._initialize_models()
                    else:
                        logger.info("DEVELOPMENT: Lazy loading enabled")
                    self._initialized = True

    def _initialize_models(self) -> None:
        start = time.time()
        self._model_load_errors.clear()

        try:
            self._load_spacy()
            self._load_transformers()
        except Exception as e:
            self._model_load_errors.append(f"Model init error: {e}")
            logger.error(f"Model initialization error: {e}")

        self._init_time = time.time() - start
        logger.info(f"Models initialized in {self._init_time:.2f}s")

    def _load_spacy(self) -> None:
        spacy_module = _lazy_import_spacy()
        if not spacy_module:
            self._model_load_errors.append("spaCy not available")
            return

        for model_name in ("en_core_web_trf", "en_core_web_sm"):
            try:
                self._models['nlp'] = spacy_module.load(model_name)
                logger.info(f"Loaded spaCy model: {model_name}")
                return
            except OSError:
                continue

        try:
            spacy_module.cli.download("en_core_web_sm")
            self._models['nlp'] = spacy_module.load("en_core_web_sm")
        except Exception as e:
            self._model_load_errors.append(f"spaCy load failed: {e}")

    def _load_transformers(self) -> None:
        st_module = _lazy_import_sentence_transformers()
        kw_module = _lazy_import_keybert()

        if not st_module or not kw_module:
            self._model_load_errors.append("sentence-transformers or keybert not available")
            return

        SentenceTransformer = st_module['SentenceTransformer']
        KeyBERT = kw_module

        model_path = os.path.join(project_root, "libs_intgr8/all-MiniLM-L12-v2")
        source = model_path if os.path.exists(model_path) else "all-MiniLM-L12-v2"

        try:
            self._models['embed_model'] = SentenceTransformer(source)
            self._models['kw_model'] = KeyBERT(source)
            self._precompute_subject_embeddings()
        except Exception as e:
            self._model_load_errors.append(f"Transformer load failed: {e}")

    def _precompute_subject_embeddings(self) -> None:
        if self._models['embed_model'] is None:
            return

        self._models['subject_embeddings'] = {
            'labels': SUBJECT_LABELS,
            'embeddings': self._models['embed_model'].encode(
                SUBJECT_LABELS, normalize_embeddings=True
            ),
        }

    def get_models(self) -> Dict[str, Any]:
        if IS_DEVELOPMENT and not self._models.get('nlp') and not self._models.get('embed_model'):
            with self._lock:
                if not self._models.get('nlp') and not self._models.get('embed_model'):
                    self._initialize_models()
        return self._models.copy()

    def health_check(self) -> Dict[str, Any]:
        models_loaded = {k: v is not None for k, v in self._models.items()}
        return {
            "status": "healthy" if any(models_loaded.values()) else "degraded",
            "models_loaded": models_loaded,
            "init_time_seconds": self._init_time,
            "errors": self._model_load_errors.copy(),
            "environment": "production" if IS_PRODUCTION else "development",
            "cache_entries": len(_result_cache._cache),
        }

    def cleanup(self) -> None:
        with self._lock:
            self._models = {
                'nlp': None, 'embed_model': None,
                'kw_model': None, 'subject_embeddings': None,
            }
            self._initialized = False
            _result_cache.clear()
            gc.collect()


_global_classifier_manager = GlobalContentClassifierManager()


def cleanup_classifier_resources():
    _global_classifier_manager.cleanup()


atexit.register(cleanup_classifier_resources)


# ─────────────────────────────────────────────
# SENTENCE SPLITTING
# ─────────────────────────────────────────────

_ABBREV_RE = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Vs|Inc|Ltd|Corp|Dept|Univ|'
    r'Approx|Est|Govt|Fig|Vol|Gen|Adm|Sgt|Cpl|Pvt|Capt|Lt|Col|Maj'
    r')\.\s*$',
    re.IGNORECASE,
)
_INITIAL_RE = re.compile(r'\b[A-Z]\.\s*$')
_DECIMAL_RE = re.compile(r'\d\.\s*$')


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, handling abbreviations and decimals."""
    raw_parts = re.split(r'(?<=[.!?])\s+(?=[A-Z\d"\'\(\[])', text)

    sentences: List[str] = []
    buffer = ""

    for part in raw_parts:
        combined = (buffer + " " + part).strip() if buffer else part

        if (_ABBREV_RE.search(combined)
                or _INITIAL_RE.search(combined)
                or _DECIMAL_RE.search(combined)):
            buffer = combined
        else:
            sentences.append(combined)
            buffer = ""

    if buffer:
        sentences.append(buffer)

    return [s.strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────
# CONTENT CLASSIFIER
# ─────────────────────────────────────────────
class ContentClassifier:
    """
    NLP-powered content metadata creator optimised for RAG pipelines.

    Extracts:
      - Named entities with frequency-based ranking
      - Key facts containing numerical data
      - Ultra-compact summaries (TextRank + fallback)
      - Keywords (KeyBERT with MMR diversity + fallback)
      - Subject classification with confidence score
      - Document type detection (Q&A, article, tutorial, ...)
      - Readability metrics (word count, vocabulary richness, ...)
      - Sentiment/tone analysis
      - Topics and themes
    """

    _instance = None
    _lock = threading.RLock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            models = _global_classifier_manager.get_models()
            self.nlp = models.get('nlp')
            self.embed_model = models.get('embed_model')
            self.kw_model = models.get('kw_model')
            self.subject_embeddings = models.get('subject_embeddings')

            self.priority_entities = [
                "PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "MONEY", "PERCENT",
            ]
            self.secondary_entities = ["LOC", "NORP", "FAC", "DATE", "TIME", "WORK_OF_ART"]

            self.max_entities = 20
            self.max_facts = 10
            self.max_keywords = 15
            self.max_topics = 10

            self._initialized = True

    # ─────────────────────────────────────────
    # TEXT PREPROCESSING
    # ─────────────────────────────────────────
    @staticmethod
    def normalize_unicode_text(text: str) -> str:
        """Normalize Unicode characters, ligatures, and smart quotes."""
        if not text:
            return text

        if "\\u" in text:
            try:
                text = text.encode("utf-8").decode("unicode_escape")
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass

        replacements = {
            '\ufb01': 'fi', '\ufb02': 'fl', '\u2013': '-', '\u2014': '--',
            '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
            '\u2022': '*', '\xa0': ' ', '\u2026': '...',
            '\u00b7': '.', '\u200b': '', '\ufeff': '',
        }
        for bad, good in replacements.items():
            text = text.replace(bad, good)

        return unicodedata.normalize('NFKC', text)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """Clean and normalize whitespace, remove citations and bare URLs."""
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\[\d+\]", "", text)
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        return text

    # ─────────────────────────────────────────
    # ENTITY EXTRACTION
    # ─────────────────────────────────────────
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities with frequency counts and smart deduplication.

        Returns dict mapping entity type to list of
        {"text": str, "count": int} ordered by frequency.
        """
        if self.nlp is None:
            return self._fallback_entity_extraction(text)

        try:
            doc = self.nlp(text[:100_000])
            allowed = set(self.priority_entities) | set(self.secondary_entities)

            raw: Dict[str, Counter] = {}
            canonical: Dict[str, Dict[str, str]] = {}

            for ent in doc.ents:
                if ent.label_ not in allowed:
                    continue
                clean = ent.text.strip()
                if len(clean) < 2:
                    continue

                label = ent.label_
                lower = clean.lower()

                if label not in raw:
                    raw[label] = Counter()
                    canonical[label] = {}

                existing = canonical[label].get(lower)
                if existing is None:
                    for seen_lower, seen_text in canonical[label].items():
                        if lower in seen_lower or seen_lower in lower:
                            longer = clean if len(clean) > len(seen_text) else seen_text
                            shorter_key = seen_lower if len(seen_lower) < len(lower) else lower
                            raw[label][longer] += raw[label].pop(seen_text, 0) + 1
                            canonical[label][seen_lower] = longer
                            canonical[label][lower] = longer
                            break
                    else:
                        canonical[label][lower] = clean
                        raw[label][clean] += 1
                else:
                    raw[label][existing] += 1

            prioritized: Dict[str, List[Dict[str, Any]]] = {}
            total = 0

            for ent_type in self.priority_entities + self.secondary_entities:
                if ent_type not in raw or total >= self.max_entities:
                    continue
                sorted_ents = raw[ent_type].most_common()
                remaining = self.max_entities - total
                selected = sorted_ents[:remaining]
                prioritized[ent_type] = [
                    {"text": name, "count": count} for name, count in selected
                ]
                total += len(selected)

            return prioritized

        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
            return self._fallback_entity_extraction(text)

    @staticmethod
    def _fallback_entity_extraction(text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Regex-based entity extraction when spaCy is unavailable."""
        entities: Dict[str, List[Dict[str, Any]]] = {}

        patterns = {
            "PERSON": r'\b([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            "ORG": r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)* (?:Inc|Corp|LLC|Ltd|Company|Corporation|Group|Industries))\b',
            "GPE": r'\b(?:in|at|from|to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            "MONEY": r'(\$[\d,]+(?:\.\d+)?(?:\s*(?:billion|million|thousand|trillion))?)',
            "PERCENT": r'(\d+(?:\.\d+)?%)',
        }

        for label, pattern in patterns.items():
            flags = re.IGNORECASE if label in ("MONEY",) else 0
            found = re.findall(pattern, text, flags)
            if found:
                counted = Counter(found)
                entities[label] = [
                    {"text": t, "count": c}
                    for t, c in counted.most_common(5)
                ]

        return entities

    # ─────────────────────────────────────────
    # FACT EXTRACTION
    # ─────────────────────────────────────────
    def extract_key_facts(self, text: str) -> List[str]:
        """Extract concise facts containing numbers, statistics, or key actions."""
        facts: List[str] = []
        sentences = split_sentences(text)

        stat_re = re.compile(
            r'\$[\d,]+(?:\.\d+)?|\d+(?:\.\d+)?%|\d[\d,]*\s*(?:billion|million|thousand|trillion)',
            re.IGNORECASE,
        )
        action_re = re.compile(
            r'\b(?:announced|revealed|reported|launched|signed|agreed|acquired|built|'
            r'created|increased|decreased|reached|exceeded|raised|invested|generated|'
            r'recorded|produced|achieved|surpassed|declined|dropped|rose|grew|fell|'
            r'expanded|contracted|doubled|tripled)\b',
            re.IGNORECASE,
        )

        for sent in sentences:
            if len(sent) < 20:
                continue

            has_stat = bool(stat_re.search(sent))
            has_number = bool(re.search(r'\d+', sent))
            has_action = bool(action_re.search(sent))

            if has_stat or (has_number and has_action):
                fact = self._compact_fact(sent)
                if fact and fact not in facts:
                    facts.append(fact)

        return facts[:self.max_facts]

    @staticmethod
    def _compact_fact(sentence: str, max_chars: int = 150) -> Optional[str]:
        """Compress a sentence into a concise fact statement."""
        sentence = sentence.strip()
        if len(sentence) <= max_chars:
            return sentence

        truncated = sentence[:max_chars]
        for punct in ('.', ';', ','):
            idx = truncated.rfind(punct)
            if idx > max_chars * 0.6:
                return truncated[:idx + 1]

        return truncated.rsplit(' ', 1)[0] + "..."

    # ─────────────────────────────────────────
    # SUMMARIZATION
    # ─────────────────────────────────────────
    def summarize(self, text: str, max_length: int = 250) -> str:
        """Generate an ultra-compact summary optimised for RAG chunk metadata."""
        if not text:
            return ""
        if len(text) <= max_length:
            return text

        cached = _result_cache.get(text, "summary")
        if cached is not None:
            return cached

        summary = self._textrank_summary(text, max_length)
        if not summary:
            summary = self._scored_sentence_summary(text, max_length)

        _result_cache.set(text, summary, "summary")
        return summary

    def _textrank_summary(self, text: str, max_length: int) -> Optional[str]:
        try:
            sumy_module = _lazy_import_sumy()
            if not sumy_module:
                return None

            parser = sumy_module['PlaintextParser'].from_string(
                text[:50_000], sumy_module['Tokenizer']("english")
            )
            summarizer = sumy_module['TextRankSummarizer']()
            num_sentences = 1 if len(text) < 1000 else 2
            result = " ".join(
                str(s) for s in summarizer(parser.document, num_sentences)
            ).strip()

            if result:
                return self._truncate_at_boundary(result, max_length)
        except Exception as e:
            logger.debug(f"TextRank summarization failed: {e}")
        return None

    def _scored_sentence_summary(self, text: str, max_length: int) -> str:
        """Select most informative sentences by weighted scoring."""
        sentences = split_sentences(text)
        candidates = [s for s in sentences if len(s) > 20]

        if not candidates:
            return self._truncate_at_boundary(text, max_length)

        scored: List[Tuple[float, int, str]] = []
        for idx, sent in enumerate(candidates[:30]):
            score = 0.0
            score += len(re.findall(r'\d+', sent)) * 2
            score += len(re.findall(r'\$[\d,]+|\d+%', sent)) * 3
            score += len(re.findall(r'\b[A-Z][a-z]+\b', sent)) * 1.5
            score -= idx * 0.3  # positional bias: earlier sentences preferred
            score += min(len(sent) / 15, 10)
            scored.append((score, idx, sent))

        scored.sort(key=lambda t: t[0], reverse=True)
        selected = sorted(scored[:2], key=lambda t: t[1])
        summary = ". ".join(s for _, _, s in selected)
        if not summary.endswith("."):
            summary += "."

        return self._truncate_at_boundary(summary, max_length)

    @staticmethod
    def _truncate_at_boundary(text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text

        truncated = text[:max_length]
        for punct in ('.', '!', '?'):
            last = truncated.rfind(punct)
            if last > max_length * 0.65:
                return truncated[:last + 1]

        return truncated.rsplit(' ', 1)[0] + "..."

    # ─────────────────────────────────────────
    # KEYWORD EXTRACTION
    # ─────────────────────────────────────────
    def extract_keywords(self, text: str, top_n: int = 15) -> List[str]:
        """Extract diverse keywords using KeyBERT with MMR, or regex fallback."""
        cached = _result_cache.get(text, f"kw:{top_n}")
        if cached is not None:
            return cached

        text = self.preprocess_text(text)
        result = self._keybert_keywords(text, top_n)
        if result is None:
            result = self._fallback_keywords(text, top_n)

        _result_cache.set(text, result, f"kw:{top_n}")
        return result

    def _keybert_keywords(self, text: str, top_n: int) -> Optional[List[str]]:
        if self.kw_model is None:
            return None
        try:
            keywords = self.kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=top_n,
                use_mmr=True,
                diversity=0.5,
            )
            return [kw for kw, _ in keywords]
        except Exception as e:
            logger.error(f"KeyBERT extraction error: {e}")
            return None

    @staticmethod
    def _fallback_keywords(text: str, top_n: int) -> List[str]:
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        freq = Counter(w for w in words if w not in STOP_WORDS and len(w) > 3)
        return [w for w, _ in freq.most_common(top_n)]

    # ─────────────────────────────────────────
    # SUBJECT CLASSIFICATION
    # ─────────────────────────────────────────
    def classify_subject(self, text: str) -> Dict[str, Any]:
        """
        Classify the primary subject/domain of the text.

        Returns {"label": str, "confidence": float, "top_3": [...]}
        """
        cached = _result_cache.get(text, "subject")
        if cached is not None:
            return cached

        result = self._embedding_subject(text)
        if result is None:
            result = self._fallback_subject_classification(text)

        _result_cache.set(text, result, "subject")
        return result

    def _embedding_subject(self, text: str) -> Optional[Dict[str, Any]]:
        if self.embed_model is None or self.subject_embeddings is None:
            return None

        try:
            st_module = _lazy_import_sentence_transformers()
            if st_module is None:
                return None

            util = st_module['util']
            labels = self.subject_embeddings['labels']
            embs = self.subject_embeddings['embeddings']

            doc_emb = self.embed_model.encode(text[:5000], normalize_embeddings=True)
            scores = util.cos_sim(doc_emb, embs)[0].tolist()

            ranked = sorted(
                zip(labels, scores), key=lambda x: x[1], reverse=True
            )

            return {
                "label": ranked[0][0],
                "confidence": round(ranked[0][1], 4),
                "top_3": [
                    {"label": lb, "confidence": round(sc, 4)}
                    for lb, sc in ranked[:3]
                ],
            }
        except Exception as e:
            logger.error(f"Embedding subject classification error: {e}")
            return None

    @staticmethod
    def _fallback_subject_classification(text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        scores: Dict[str, int] = {}

        for domain, keywords in DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return {"label": "General", "confidence": 0.0, "top_3": []}

        total = sum(scores.values())
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return {
            "label": ranked[0][0],
            "confidence": round(ranked[0][1] / total, 4) if total else 0.0,
            "top_3": [
                {"label": lb, "confidence": round(sc / total, 4)}
                for lb, sc in ranked[:3]
            ],
        }

    # ─────────────────────────────────────────
    # DOCUMENT TYPE DETECTION
    # ─────────────────────────────────────────
    @staticmethod
    def detect_document_type(text: str) -> Dict[str, Any]:
        """
        Detect document type (question_answer, tutorial, report, legal,
        narrative, technical, news, article).

        Returns {"type": str, "confidence": float, "scores": {...}}
        """
        text_sample = text[:10_000]
        scores: Dict[str, int] = {}

        for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
            total_matches = 0
            for pattern in patterns:
                total_matches += len(re.findall(pattern, text_sample, re.IGNORECASE | re.MULTILINE))
            if total_matches > 0:
                scores[doc_type] = total_matches

        if not scores:
            return {"type": "article", "confidence": 0.3, "scores": {}}

        best_type = max(scores, key=scores.get)
        total = sum(scores.values())
        confidence = round(scores[best_type] / total, 4) if total else 0.0

        return {
            "type": best_type,
            "confidence": min(confidence, 1.0),
            "scores": {k: round(v / total, 4) for k, v in scores.items()},
        }

    # ─────────────────────────────────────────
    # TEMPORAL MARKERS
    # ─────────────────────────────────────────
    @staticmethod
    def extract_temporal_markers(text: str) -> List[str]:
        """
        Extract date and time references for temporal context metadata.

        Captures explicit dates, relative time expressions, fiscal periods,
        and year references using a comprehensive multi-pattern scan.
        """
        found: List[str] = []
        seen: Set[str] = set()
        for pattern in TEMPORAL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                val = match.group().strip()
                key = val.lower()
                if key not in seen:
                    seen.add(key)
                    found.append(val)
        return found[:20]

    # ─────────────────────────────────────────
    # STRUCTURAL FEATURES
    # ─────────────────────────────────────────
    @staticmethod
    def detect_structural_features(text: str) -> Dict[str, Any]:
        """
        Detect structural and formatting elements present in the document.

        Returns boolean flags and counts for: code blocks, bullet/numbered
        lists, tables, section headers, questions, citations, mathematical
        formulas, URLs, and emails. Also returns paragraph and line counts.
        """
        code_re = re.compile(
            r'```[\w]*\n|`[^`\n]+`|\bdef\s+\w+\s*\(|\bclass\s+\w+[\s:(]'
            r'|\bfunction\s+\w+\s*\(|\bimport\s+[\w.]+|\bfrom\s+\w+\s+import\b',
            re.MULTILINE,
        )
        list_re = re.compile(
            r'^\s*[-*•▪◦]\s+\S|^\s*\d+[.)]\s+\S|^\s*[a-z][.)]\s+\S',
            re.MULTILINE,
        )
        table_re = re.compile(r'\|.+\|.+\||\+-{2,}\+|\t\w+(?:\t\w+)+', re.MULTILINE)
        header_re = re.compile(
            r'^#{1,6}\s+\S|^[A-Z][A-Z\s\d]{3,}:?\s*$|^={3,}\s*$|^-{3,}\s*$',
            re.MULTILINE,
        )
        question_re = re.compile(r'\?\s*$|^\s*Q\d*[.:)]\s', re.MULTILINE)
        citation_re = re.compile(r'\[\d+\]|\(\w+,\s+\d{4}\)|\b(?:et al\.?|ibid\.?)\b')
        formula_re = re.compile(
            r'\$[^$\n]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]'
            r'|[∑∫∂∇±×÷≤≥≠≈∞αβγδεζηθ]',
        )
        url_re = re.compile(r'https?://\S+|www\.\S+')
        email_re = re.compile(r'\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b')

        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if len(p.strip()) > 15]
        lines = text.split('\n')

        return {
            "has_code": bool(code_re.search(text)),
            "has_lists": bool(list_re.search(text)),
            "has_tables": bool(table_re.search(text)),
            "has_headers": bool(header_re.search(text)),
            "has_questions": bool(question_re.search(text)),
            "has_citations": bool(citation_re.search(text)),
            "has_formulas": bool(formula_re.search(text)),
            "has_urls": bool(url_re.search(text)),
            "has_emails": bool(email_re.search(text)),
            "paragraph_count": len(paragraphs),
            "line_count": len(lines),
            "avg_paragraph_length": round(
                sum(len(p) for p in paragraphs) / max(len(paragraphs), 1)
            ),
        }

    # ─────────────────────────────────────────
    # NAMED CONCEPTS
    # ─────────────────────────────────────────
    def extract_named_concepts(self, text: str, top_n: int = 15) -> List[str]:
        """
        Extract noun phrases and named concepts from text.

        Uses spaCy noun chunks when available; falls back to regex extraction
        of capitalized multi-word phrases and high-frequency content words.
        """
        if self.nlp is not None:
            try:
                doc = self.nlp(text[:50_000])
                freq: Counter = Counter()
                for chunk in doc.noun_chunks:
                    if len(chunk) > 1:
                        phrase = chunk.text.strip().lower()
                    else:
                        phrase = chunk.root.lemma_.lower()
                    phrase = re.sub(r'\s+', ' ', phrase).strip()
                    if len(phrase) > 3 and phrase not in STOP_WORDS:
                        freq[phrase] += 1
                return [phrase for phrase, _ in freq.most_common(top_n)]
            except Exception as e:
                logger.debug(f"spaCy noun-chunk extraction failed: {e}")

        # Regex fallback: capitalized multi-word phrases + frequent content words
        cap_phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        content_words = re.findall(r'\b[a-zA-Z]{5,}\b', text.lower())

        freq: Counter = Counter()
        for phrase in cap_phrases:
            freq[phrase.lower()] += 2
        for word in content_words:
            if word not in STOP_WORDS:
                freq[word] += 1

        return [phrase for phrase, _ in freq.most_common(top_n)]

    # ─────────────────────────────────────────
    # CONTENT FINGERPRINT
    # ─────────────────────────────────────────
    @staticmethod
    def compute_content_fingerprint(text: str) -> str:
        """Compute a short SHA-256 fingerprint of the text for deduplication."""
        normalized = re.sub(r'\s+', ' ', text.lower().strip())[:3000]
        return hashlib.sha256(normalized.encode('utf-8', errors='replace')).hexdigest()[:16]

    # ─────────────────────────────────────────
    # MULTI-LABEL DOCUMENT TYPE
    # ─────────────────────────────────────────
    def detect_document_types_multilabel(self, text: str) -> List[Dict[str, Any]]:
        """
        Return all detected document types with confidence scores.

        Unlike detect_document_type which returns only the single best type,
        this method returns every matched type ranked by confidence, enabling
        multi-label classification for hybrid documents.
        """
        text_sample = text[:10_000]
        scores: Dict[str, int] = {}

        for doc_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
            total_matches = sum(
                len(re.findall(p, text_sample, re.IGNORECASE | re.MULTILINE))
                for p in patterns
            )
            if total_matches > 0:
                scores[doc_type] = total_matches

        if not scores:
            return [{"type": "article", "confidence": 0.3}]

        total = sum(scores.values())
        return sorted(
            [{"type": t, "confidence": round(c / total, 4)} for t, c in scores.items()],
            key=lambda x: -x["confidence"],
        )

    # ─────────────────────────────────────────
    # READABILITY METRICS
    # ─────────────────────────────────────────
    @staticmethod
    def compute_readability(text: str) -> Dict[str, Any]:
        """
        Compute readability and complexity metrics for RAG ranking.

        Returns word_count, sentence_count, avg_sentence_length,
        vocabulary_richness, syllable_density, and estimated reading level.
        """
        sentences = split_sentences(text)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        word_count = len(words)
        sentence_count = max(len(sentences), 1)

        if word_count == 0:
            return {
                "word_count": 0,
                "sentence_count": sentence_count,
                "avg_sentence_length": 0.0,
                "vocabulary_richness": 0.0,
                "avg_word_length": 0.0,
                "information_density": 0.0,
                "reading_level": "unknown",
            }

        avg_sent_len = round(word_count / sentence_count, 2)
        unique_words = set(w.lower() for w in words)
        vocab_richness = round(len(unique_words) / word_count, 4)
        avg_word_len = round(sum(len(w) for w in words) / word_count, 2)

        number_count = len(re.findall(r'\d+', text))
        proper_noun_count = len(re.findall(r'\b[A-Z][a-z]+\b', text))
        info_density = round(
            (number_count + proper_noun_count) / max(sentence_count, 1), 4
        )

        if avg_sent_len < 12 and avg_word_len < 4.5:
            reading_level = "elementary"
        elif avg_sent_len < 18 and avg_word_len < 5.5:
            reading_level = "intermediate"
        elif avg_sent_len < 25:
            reading_level = "advanced"
        else:
            reading_level = "academic"

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sent_len,
            "vocabulary_richness": vocab_richness,
            "avg_word_length": avg_word_len,
            "information_density": info_density,
            "reading_level": reading_level,
        }

    # ─────────────────────────────────────────
    # SENTIMENT / TONE ANALYSIS
    # ─────────────────────────────────────────
    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, Any]:
        """
        Lightweight lexicon-based sentiment analysis.

        Returns {"sentiment": positive|negative|neutral,
                 "polarity": float -1..1, "subjectivity": float 0..1}
        """
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if not words:
            return {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0}

        pos_count = sum(1 for w in words if w in SENTIMENT_POSITIVE)
        neg_count = sum(1 for w in words if w in SENTIMENT_NEGATIVE)
        total_sentiment = pos_count + neg_count

        polarity = 0.0
        if total_sentiment > 0:
            polarity = round((pos_count - neg_count) / total_sentiment, 4)

        subjectivity = round(min(total_sentiment / max(len(words), 1) * 10, 1.0), 4)

        if polarity > 0.15:
            sentiment = "positive"
        elif polarity < -0.15:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "polarity": polarity,
            "subjectivity": subjectivity,
        }

    # ─────────────────────────────────────────
    # LANGUAGE DETECTION
    # ─────────────────────────────────────────
    @staticmethod
    def detect_language(text: str) -> str:
        """
        Basic language detection by character-set heuristic.
        Returns ISO 639-1 code or 'unknown'.
        """
        sample = text[:2000]

        if re.search(r'[\u4e00-\u9fff]', sample):
            return "zh"
        if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', sample):
            return "ja"
        if re.search(r'[\uac00-\ud7af]', sample):
            return "ko"
        if re.search(r'[\u0600-\u06ff]', sample):
            return "ar"
        if re.search(r'[\u0900-\u097f]', sample):
            return "hi"
        if re.search(r'[\u0400-\u04ff]', sample):
            return "ru"

        ascii_ratio = sum(1 for c in sample if c.isascii()) / max(len(sample), 1)
        if ascii_ratio > 0.85:
            return "en"

        return "unknown"

    # ─────────────────────────────────────────
    # TOPIC EXTRACTION
    # ─────────────────────────────────────────
    def extract_topics_and_themes(self, text: str) -> List[str]:
        """Extract de-duplicated topics combining subject, keywords, and patterns."""
        topics: List[str] = []

        subject = self.classify_subject(text)
        if subject["label"] != "General":
            topics.append(subject["label"])

        keywords = self.extract_keywords(text, top_n=10)
        topics.extend(keywords[:7])

        topic_patterns = re.findall(
            r'\b(?:about|regarding|concerning|focusing on|related to)\s+'
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            text,
            re.IGNORECASE,
        )
        topics.extend(t for t in topic_patterns if len(t) > 3)

        seen: Set[str] = set()
        unique: List[str] = []
        for topic in topics:
            key = topic.lower().strip()
            if key not in seen and len(key) > 1:
                seen.add(key)
                unique.append(topic)

        return unique[:self.max_topics]

    # ─────────────────────────────────────────
    # MAIN ANALYSIS
    # ─────────────────────────────────────────
    def analyze(
        self,
        corpus: Any,
        url: Optional[str] = None,
        max_length: Optional[int] = None,
    ) -> Any:
        """
        Full metadata extraction for RAG systems.

        Args:
            corpus:     Text to analyse.
            url:        Optional source URL (included in output).
            max_length: If provided, returns a summary string instead of
                        full metadata dict (backward-compatible shortcut).

        Returns:
            dict with all extracted metadata, or str if max_length given.
        """
        try:
            if isinstance(url, int) and max_length is None:
                max_length = url
                url = None

            corpus = str(corpus)
            corpus = self.normalize_unicode_text(corpus)
            corpus = self.preprocess_text(corpus)

            if max_length is not None:
                return self.summarize(corpus, max_length) if corpus else None

            if not corpus or len(corpus.strip()) < 10:
                return self._empty_result(corpus, url)

            subject = self.classify_subject(corpus)

            result = {
                # Subject / domain classification
                "subject": subject["label"],
                "subject_confidence": subject["confidence"],
                "subject_top_3": subject["top_3"],
                # Document type (single best + full multi-label)
                "document_type": self.detect_document_type(corpus),
                "document_types": self.detect_document_types_multilabel(corpus),
                # Language
                "language": self.detect_language(corpus),
                # NLP extractions
                "entities": self.extract_entities(corpus),
                "keywords": self.extract_keywords(corpus, self.max_keywords),
                "named_concepts": self.extract_named_concepts(corpus),
                "topics_themes": self.extract_topics_and_themes(corpus),
                "temporal_markers": self.extract_temporal_markers(corpus),
                # Text content
                "summary": self.summarize(corpus, 300),
                "key_facts": self.extract_key_facts(corpus),
                # Structure & style
                "structural_features": self.detect_structural_features(corpus),
                "readability": self.compute_readability(corpus),
                "sentiment": self.analyze_sentiment(corpus),
                # Deduplication
                "content_fingerprint": self.compute_content_fingerprint(corpus),
            }

            if url:
                result["url"] = url

            return result

        except Exception as e:
            logger.error(f"Analysis error: {e}", exc_info=True)
            return self._error_result(corpus, url, str(e))

    def analyze_batch(
        self,
        texts: List[str],
        urls: Optional[List[Optional[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Analyse multiple texts in sequence (cache-friendly).

        Args:
            texts: List of text corpora.
            urls:  Optional parallel list of URLs.

        Returns:
            List of metadata dicts, one per input text.
        """
        urls = urls or [None] * len(texts)
        return [
            self.analyze(text, url=url) for text, url in zip(texts, urls)
        ]

    # ─────────────────────────────────────────
    # RESULT TEMPLATES
    # ─────────────────────────────────────────
    @staticmethod
    def _empty_result(corpus: str, url: Optional[str]) -> Dict[str, Any]:
        _empty_structural: Dict[str, Any] = {
            "has_code": False, "has_lists": False, "has_tables": False,
            "has_headers": False, "has_questions": False, "has_citations": False,
            "has_formulas": False, "has_urls": False, "has_emails": False,
            "paragraph_count": 0, "line_count": 0, "avg_paragraph_length": 0,
        }
        return {
            "subject": "Unknown",
            "subject_confidence": 0.0,
            "subject_top_3": [],
            "document_type": {"type": "unknown", "confidence": 0.0, "scores": {}},
            "document_types": [],
            "language": "unknown",
            "entities": {},
            "keywords": [],
            "named_concepts": [],
            "summary": corpus[:200] if corpus else "",
            "key_facts": [],
            "topics_themes": [],
            "temporal_markers": [],
            "structural_features": _empty_structural,
            "readability": {
                "word_count": 0, "sentence_count": 0,
                "avg_sentence_length": 0.0, "vocabulary_richness": 0.0,
                "avg_word_length": 0.0, "information_density": 0.0,
                "reading_level": "unknown",
            },
            "sentiment": {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0},
            "content_fingerprint": "",
            "url": url,
        }

    @staticmethod
    def _error_result(corpus: str, url: Optional[str], error: str) -> Dict[str, Any]:
        _empty_structural: Dict[str, Any] = {
            "has_code": False, "has_lists": False, "has_tables": False,
            "has_headers": False, "has_questions": False, "has_citations": False,
            "has_formulas": False, "has_urls": False, "has_emails": False,
            "paragraph_count": 0, "line_count": 0, "avg_paragraph_length": 0,
        }
        return {
            "subject": "Error",
            "subject_confidence": 0.0,
            "subject_top_3": [],
            "document_type": {"type": "unknown", "confidence": 0.0, "scores": {}},
            "document_types": [],
            "language": "unknown",
            "entities": {},
            "keywords": [],
            "named_concepts": [],
            "summary": corpus[:200] if corpus else "Error processing",
            "key_facts": [],
            "topics_themes": [],
            "temporal_markers": [],
            "structural_features": _empty_structural,
            "readability": {
                "word_count": 0, "sentence_count": 0,
                "avg_sentence_length": 0.0, "vocabulary_richness": 0.0,
                "avg_word_length": 0.0, "information_density": 0.0,
                "reading_level": "unknown",
            },
            "sentiment": {"sentiment": "neutral", "polarity": 0.0, "subjectivity": 0.0},
            "content_fingerprint": "",
            "url": url,
            "error": error,
        }


# ─────────────────────────────────────────────
# FACTORY / HEALTH
# ─────────────────────────────────────────────

def get_content_classifier() -> ContentClassifier:
    """Get the ContentClassifier singleton."""
    return ContentClassifier()


def health_check_classifier() -> Dict[str, Any]:
    """Return model manager health report."""
    return _global_classifier_manager.health_check()


# ─────────────────────────────────────────────
# QUICK DEMO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sample = """
Questions and Answers
1. What category does iron ore belong to among the following types of resources?
(a) Renewable
(b) Flow
(c) Biotic
(d) Non-renewable
Ans: (d) Non-renewable

2. Explain why resources depend on human activities.
Ans: Resources exist because humans use and manage them for their needs. Human activities like farming, mining, and manufacturing create resources by extracting or transforming natural materials. Without human intervention, many resources wouldn't be accessible or usable.

3. How are resources classified based on how much they can be used? Give two examples of each.
Ans: Resources classified based on:
Renewable Resources:
These resources can be replaced naturally or through human efforts, like solar energy and forests.
They are continually available if managed sustainably.
Non-renewable Resources:
These resources are finite and cannot be replaced once used up, such as fossil fuels (coal, oil) and minerals.
Their availability decreases over time with increasing human consumption.

4. Which type of soil is known for growing cotton?
Ans: The soil known as "black soil" or "regur soil" is ideal for growing cotton.

This soil type is rich in nutrients like calcium, magnesium, and potassium, which are beneficial for cotton cultivation.

5. What is the primary reason for land degradation in Punjab?
(a) Intensive cultivation
(c) Over irrigation
(b) Deforestation
(d) Overgrazing
Ans: (c) Over irrigation

6. Give two examples of renewable resources.
Ans: Examples of renewable resources include solar energy, which comes from the sun and can be harnessed through solar panels, and wind energy, which is generated using wind turbines to produce electricity.

7. Name three states having black soil and the crop which is mainly grown in it.
Ans: Black soil is found in Maharashtra, Gujarat, and Madhya Pradesh in India. Cotton is the main crop grown in these states because black soil is rich in nutrients and retains moisture well, making it suitable for cotton cultivation.

8. What steps can be taken to control soil erosion in hilly areas?
Ans: The steps that can be taken to control soil erosion in hilly areas are:

To control soil erosion in hilly areas:
Planting trees and grass to stabilise the soil.
Building terraces or contour ploughing to slow down water flow.
Using mulch or cover crops to protect soil from rain impact.
Constructing check dams or retaining walls to reduce runoff.
Avoid overgrazing and deforestation to maintain vegetation cover.
Implementing proper land management practices and erosion control measures.

9. How are natural resources important for humans?
Ans: These are the ways natural resources are important for humans:

Natural resources are vital because they provide raw materials for industries.
They support agriculture, providing fertile land and water for crops.
Natural resources like minerals and fuels are essential for energy production.
They contribute to biodiversity, supporting ecosystems and wildlife.
Natural resources also offer recreational opportunities for people.

10. What do you understand about “sustainable economic development”?
Ans: Sustainable economic development means growing the economy while preserving resources for future generations.

It involves using resources efficiently without depleting them.
Sustainable development promotes social equity and environmental protection.
It aims for long-term prosperity without compromising the ability of future generations to meet their needs.
Achieving sustainable economic development requires balancing economic growth with environmental and social considerations.

11. What is Agenda 21?
Ans: Agenda 21 is a global action plan adopted at the United Nations Earth Summit in 1992.

It outlines strategies for sustainable development across various sectors like agriculture, forestry, and industry.
Agenda 21 aims to address environmental issues and promote sustainable practices worldwide.
It encourages international cooperation to tackle challenges such as climate change and biodiversity loss.
Agenda 21 emphasises the role of governments, businesses, and individuals in achieving sustainable development goals.

12. What type of soil is found in the river deltas of the eastern coast? Give three main features of this type of soil.
Ans: The  type of soil found in the river deltas of the eastern coast are:

The soil found in river deltas of the eastern coast is alluvial.
It is rich in nutrients like nitrogen and phosphorus, making it fertile for agriculture.
Alluvial soil is well-drained and retains moisture, which is beneficial for crop growth.
It is easy to cultivate and supports the cultivation of crops like rice, wheat, and sugarcane.
Alluvial soil is formed by the deposition of silt and clay carried by rivers, making it ideal for agricultural productivity.

13. What are biotic and abiotic resources? Give some examples.
Ans: Biotic resources are derived from living organisms, such as forests, animals, and fish, which provide food, fuel, and other essentials. Abiotic resources, like minerals, water, and air, are non-living and essential for industries and daily life. Examples include:

Biotic: Trees for timber, animals for food, fish for fisheries.
Abiotic: Minerals like iron ore, water for drinking, air for breathing.

14. How has technical and economic development led to increased resource consumption?
Ans: Technological advancements and economic growth have driven higher resource consumption by:

Increased Efficiency: Advanced technologies require more resources to manufacture and operate effectively.
Growing Demand: Economic development leads to higher demand for raw materials and energy to sustain industries and lifestyles.

15. Explain the 3 stages of resource planning.
Ans: Resource planning involves:

Assessment: Evaluating current resources, their availability, and future needs based on population growth and development goals.
Planning: Formulating strategies and policies to manage resources sustainably, considering conservation and efficiency.
Implementation: Putting plans into action through regulations, incentives, and infrastructure development to ensure resource use meets present and future demands.

16. Who did Gandhiji hold responsible for global resource depletion?
Ans: Gandhiji attributed global resource depletion to industrial nations and their excessive consumption practices:

Industrial Nations: Developed countries using resources at high rates for industrialization and economic growth.
Environmental Impact: This results in environmental degradation and resource scarcity globally, impacting future generations.

17. Explain soil erosion and give the steps that should be taken to control soil erosion.
Ans: Soil erosion is the process where soil is worn away by wind, water, or human activity. It's harmful because it reduces soil fertility and damages ecosystems. Steps to control it include:

Planting trees and grass to hold soil in place.
Building terraces to slow water flow.
Using cover crops to protect soil during periods of no crop growth.

18. What is Laterite soil?
Ans: Laterite soil is a type of soil found in tropical regions. It's rich in iron oxide and aluminium oxide, making it red or orange. This soil is poor in nutrients and not suitable for agriculture without proper treatment.

19. Why has the land under forests not increased much from 1960-61?
Ans: The land under forests has not increased significantly since 1960-61 due to:

Urbanisation and industrialization lead to deforestation.
Agricultural expansion and infrastructure development reducing forest areas.
Lack of effective conservation policies and enforcement to protect forest lands.

20. State the geographical factors that are responsible for the evolution of black soil. Why is it considered the most suitable for growing cotton?
Ans: Black soil, formed from the weathering of basaltic rock under hot and dry conditions, contains minerals like calcium and magnesium, making it rich in nutrients. This soil's moisture-retaining capacity and fertility make it ideal for cotton cultivation, as cotton plants require well-drained soil with good nutrient availability.

Black soil forms from weathered basaltic rock.
It contains nutrients like calcium and magnesium.
Its ability to retain moisture benefits cotton growth.
This soil type is well-suited for hot, dry climates.

21. Write three physical and three human factors which determine the use of land.
Ans: Land use is influenced by natural and human-related factors:

Soil fertility impacts agricultural productivity.
Climate dictates what crops can thrive in a region.
Topography influences land suitability for construction or farming.
Economic activities such as agriculture or industry.
Urbanisation changes land into cities and towns.
Government regulations on zoning and conservation efforts.

22. Write four institutional efforts made at the global level for ‘resource conservation’.
Ans: Global efforts focus on sustainable resource management:

The United Nations' Sustainable Development Goals (SDGs) promote responsible resource use worldwide.
International agreements like the Paris Agreement aim to reduce greenhouse gas emissions and combat climate change.
Environmental organisations monitor and protect biodiversity to conserve natural resources.
Research institutions develop technologies to improve resource efficiency and reduce environmental impact.

23. Distinguish between the following:
a) Potential and Developed Resources
b) Stock and Reserves
Ans: 
a) Potential and Developed Resources:
Potential resources refer to those that exist in a region but are not currently exploited due to technological limitations or economic constraints. They have the potential for future development and utilisation. In contrast, developed resources are those that have been identified, explored, and exploited to meet human needs. These resources are actively extracted and utilised.

In detail:
Potential resources: These are reserves that exist in a region but are not yet tapped due to technological or economic limitations.
Developed resources: These are reserves that have been identified, explored, and exploited to meet current human needs and demands.
b) Stock and Reserves:
Stock represents the total quantity of a resource that exists at a given point in time. It includes all known and unknown quantities of the resource. Reserves, on the other hand, refer to the portion of the stock that has been identified and is economically viable for extraction under current technological and economic conditions.
In detail:
Stock: The total amount of a resource available, including both known and potential quantities.
Reserves: The part of the stock that is identified, accessible, and economically feasible to extract with existing technology and economic conditions.

24. Why is sustainable development important?
Ans: Sustainable development ensures that present needs are met without compromising the ability of future generations to meet their own needs. It balances economic growth, environmental protection, and social well-being to create a better future for all.

Sustainable development promotes responsible use of resources.
It helps in preserving biodiversity and ecosystems.
It ensures equitable distribution of resources and opportunities.
It reduces environmental degradation and pollution.
It supports long-term economic stability and social progress.

25. What are natural resources?
Ans: Natural resources are materials or substances found in nature that have economic value. They are essential for the functioning of ecosystems and human survival and development.

Examples include water, air, minerals, forests, and wildlife.
They can be renewable (like sunlight and wind) or nonrenewable (like fossil fuels).
Natural resources are used to produce goods and services for human consumption.
They are vital for agriculture, industry, and energy production.
Conservation of natural resources is crucial to ensure their sustainable use for future generations.

26. What are the measures to conserve resources?
Ans: Conservation of resources involves using resources wisely and sustainably to minimise waste and environmental impact.

Reduce, reuse, and recycle to minimise resource consumption and waste generation.
Adopt efficient technologies and practices for energy and water conservation.
Protect natural habitats and biodiversity through conservation efforts.
Promote sustainable agriculture and forestry practices.
Implement policies and regulations to prevent overexploitation of natural resources.

These measures help ensure that natural resources are preserved for future generations while supporting economic development and environmental sustainability.
    """

    classifier = get_content_classifier()
    result = classifier.analyze(sample)
    print(json.dumps(result, indent=2, default=str))
