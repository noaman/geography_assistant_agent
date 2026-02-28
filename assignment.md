Problem statement & Rubrics
Assignment-1: Gen. AI- Pre-Trained Models
Total Marks: 25
 
Problem Statement
 
Listed below are 10 jobs that are common today. Under each job, one specific task, which is part of that job, is listed. 
Legal assistant
* For a new case, find connected / similar past cases.
Bank Executive
* Answer customers’ account-related questions. 
Telemarketer
* Do a natural conversation which covers all points from a prepared script.
Insurance Underwriter
* Review submitted documents. List information gaps.  
Secondary School Teacher
* Prepare how to teach a new lesson.
Sales copy writer
* Research existing products to find key points to pitch for a new product.
HR executive
* Do an informed salary negotiation with a recruit. 
Graphic designer
* Ask clarification questions about a business brief.
Receptionist
* Pre-plan and ensure that all appointments happen on time.
Professional swimming coach
* Choose the right collection of videos for an athlete to watch.
For these jobs, we list below a Knowledge Base (KB), an adequately large set of carefully curated historical documents. The primary purpose of a KB is to contextually help new (human) employees to do their job better. New employees are expected to use a RAG-based GenAI tool that you will design, and ask questions like the Sample Query below, to seek help.

You will be responsible for creating your own Knowledge Base (KB). Feel free to use any open-source KB as a reference or starting point.
Legal assistant
* KB: logs of past search sessions and outcomes
* Sample Query: “I have no idea how to find related cases for this filing. Can you help me get started, with some search phrases?”
Bank Executive
* KB: Transcripts of past customer conversations by expert executives
* Sample Query: “What questions do new home loan customers commonly ask in this city?”
Telemarketer
* KB: Transcripts of past calls by expert callers.
* Sample Query: “How can I justify additional shipping & handling charges?” 
Insurance Underwriter
* KB: Past claims and corresponding decisions by expert underwriters
* Sample query: “What parts of claims like this one should be rejected?”
Secondary School Teacher
* KB: Comprehensive reference materials related to lessons
* Sample Query: “What other interesting things can I say about the San Andreas Fault?” 
Sales copy writer
* KB: past sales copies and data on their market performance
* Sample Query: “Can I add something to this sales copy to increase its chances of success?”
HR executive
* KB: Transcripts of past salary negotiations by expert executives
* Sample Query: “Is it even possible to convince a recruit to accept 10% less static pay? Show me how.”
Graphic designer
* KB: Past business briefs and corresponding clarification questions asked by expert designers
* Sample Query: “Please help me ask the right questions on this business brief.”
Receptionist
* KB: Details of past appointments which failed to happen on time despite pre-planning
* Sample query: “Which of this week’s appointments are the most tricky?”
Professional swimming coach
* KB: Logs of past video suggestions, how much swimmers watched them, performance impact
* Sample Query: “Am I missing any must-watch videos for this athlete profile?”
Please pick any ONE job of your choice. Focus on the task given for it. English is the medium. 
Do you think Generative AI can help with this task -
(a) Fully (100%),
(b) Mostly (80% or more),
(c) Partly (at least 40%) or
(d) Hardly (less than 20%)? 

What part of the task can GenAI do and NOT do? 

Can you provide how one should go about building the needed GenAI solution – would it be from scratch, or by finetuning / refining an existing model, or by prompting, using RAG? What problems, if any, do you foresee if one relies on GenAI to accomplish (parts of) this task, as opposed to traditionally relying on a human?   

Your assignment is to conceptually design and architect prompts and a RAG-based GenAI solution. You are at liberty to make any reasonable assumptions about the format of the KB documents, about sample queries, etc. The GenAI solution you present should justifiably be able to answer your sample queries, from your KB. We encourage you to make unambiguous choices for all tools and components that your solution needs, to the best of your knowledge.

Please feel free to leverage variety of prompts and RAGs, or any other prompts, RAG-related ideas from the literature, in your solution design. It is important to mention all known limitations of your solution (e.g., TCO, Response Time, etc).

Your answer can be a lucid note addressing the above points and more, 1-7 A4 pages with normal formatting. You can add screenshots of the solution/codes for another 3 pages if applicable. Content after 7+3 pages will simply be ignored during grading! 

Score distribution for Assignment 1: (20% each):
Detailing of KB and Sample Query format 
Overall GenAI solution design 
Prompts and RAG effectiveness enhancement 
Identification of solution limitations 
Efforts to try out the solution in Azure OpenAI / ChatGPT / any other platform

Criterion
(Out of 5)	Good
(5 or 4)	Satisfactory
(3 or 2)	Needs Improvement
(1)	Did not address
(0)
Detailing of KB and Sample Query format	Clear description of the KB and queries.	The general background is presented but the specific KB / Query is not clearly stated	Neither the KB nor query specifics of the analysis is clearly stated	 
Overall GenAI solution design	The approach is clearly described. In addition, the method of estimating the success of the approach is also clearly stated.	Either the description of the approach or the method of evaluating its success is not clear	Neither the description of the approach not the evaluation methods is described clearly.	 
RAG effectiveness enhancement	Technically correct solution	Some gaps are there in the technical evaluation	The solution presented is not technically correct.	 
Identification of solution limitations	Clearly stated limitation	Satisfactory	Basic level	 
Efforts to try out the solution in Azure OpenAI / ChatGPT platform	Significant	Medium	Basic solution	 
