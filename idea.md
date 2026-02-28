
Buit a KB with Grade 10 Geography text book

The KB was built using ChromaDB as a vector DB and NLP based meta data extractor 


- The agents use a rag_query tool which sends enriched querires to the vector DB in the following sequence

a) user sends a query
b) Before the agent call, the code fires the query to the underlying KB and gets only the META data of the top 5 docuemnts that match. This meta data has the summary, leywords and core concepts covered in each document
c)This meta data and the user input is then sent to the Agent who now has a much deeper understadning of the KB and it then formultaes 3 to 5 enriched queries to source the most relebvatn documents
d)oince the docuemnts are sourced then the agent follows the given instrucuotjs and reponsed to the user.





Ways to Improve further :
------------------------
- The RAG meta data is genreic, for specific needs we can have the capters isolated as key node for all queries to further enrich the document sourcing

- Preprocess documents before ingesiton to further claissify into aspects like cocnept notes, important questions, images etc



Sample Questions :
----------------
I will be. covering Soil Resources today in my class... Build me an exhaustive study note where I can easily catch up on core concepts, do some class activities, give aligned homework. Ensure its comprehensive to help me cover the topic fully


Build me detailed study note to teach the Climate chapter in class. I want the detailed notes on core concepts, some nice quizzes, activities and also important questions for the board 

I want to teach "Waste management" to my class - I want to make it interactive with activities - I want to use mind maps, flash cards, quizzes and also leave the class with relevant homework


AGENT Details
-------------

Systenmt Promot :
You are a Grade 10 Teaching Assistant. Your role is to help teachers prepare lecture materials by generating study notes, mind maps, question sets, anecdotes, and other teaching aids based on the request you receive. Avoid providing personal opinions, unrelated content, or any copyrighted material without proper attribution. Respond in a clear, organized manner using headings and bullet points where appropriate, and keep the tone professional and supportive.

Instructiosn :
1. Read the teacher's request carefully and identify the subject, topic, and type(s) of material needed.,2. If the request is ambiguous, ask for clarification before proceeding.,3. Query the internal knowledge base for accurate curriculum information, examples, and relevant facts.,4. Generate the requested material(s): study notes, mind maps, question sets, anecdotes, or teaching aids.,5. Format the output with clear headings; use bullet points or numbered lists for items.,6. Perform a self‑check: ensure the content is accurate, relevant to Grade 10, and free of errors.,7. Deliver the final response to the teacher.



The Knowldebase content:
-------------------
georgaphy text book having following chaptets and concepts

Part I : MAP WORK
1. Interpretation of
Topographical Maps 1
u Self Assessment Paper-1 18
2. Location, Extent and
Physical Features 19
u Self Assessment Paper-2 39
Part II : GEOGRAPHY OF INDIA
-
-
17
18
-
-
38
39
8. Agriculture 131
u Self Assessment Paper-8 163
9. Manufacturing Industries 164
u Self Assessment Paper-9 186
10. Transport 187
u Self Assessment Paper-10 202
11. Waste Management 203
u Self Assessment Paper-11 222
u Practice Paper-1 223
u Practice Paper-2 227
-
-
-
-
-
-
-
-
-
-
162
163
185
186
201
202
221
222
226
231
3. Climate 40
u Self Assessment Paper-3 55
4. Soil Resources 56
u Self Assessment Paper-4 71
5. Natural Vegetation 72
u Self Assessment Paper-5 89
6. Water Resources 90
u Self Assessment Paper-6 105
7. Mineral and Energy Resources 106
u Self Assessment Paper-7 130
-
-
-
-
-
-
-
-
-
-
54
55
70
71
88
89
104
105
129
130