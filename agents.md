Hellu
I am talking to you through Cline Desktop

The address maybe here: `D:\Cline`

This is the information about how my laptop can run LLMs: `D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\Local_LLM_runable`

This is the information about the system, it is outdated, but I think it is quite useful: `D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\ClineClean`

This file defines the objectives, constraints, invariants, and decision boundaries governing your work. It is not an implementation manual.
You are expected to exercise technical judgment and choose appropriate implementations independently, provided that your decisions remain within the constraints defined here.

# 0. Environment
- Create the .venv...or other kinds of environment in this project only, not in C

- You are free to install anything you need during writing code, no need to ask me for permission, as long as...in this folder

- You can use Docker if needed, but prioritize simplicity

- You may generate many artifacts such as temp files
+ Please, create them only in this project, not in the C disk
+ Then delete the temp files
+ Quite recommend create a folder call "agent-work" and all tmp files are there

# 1. You - coding agent, supports me in paper

- Overview about the paper: `D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion\paperOverview.md`

# 2. Parallel SRE and Coding
- About 10 tool calls, please check resources: RAM, CPU, GPU, processes and network

- Use timestamps to monitor execution timing and detect anomalies, not merely for logging.
+ Observe the time before and after tool calls, and compare elapsed wall-clock time against the expected duration.
+ Example: Two consecutive commands each take 1 second to execute, but their timestamps are 3 minute apart. This may indicate an anomaly worth investigating.

- Before extensive resources consuming (or extensive time consuming): Please check the resources again

- This is to help you understand the device status

- And before any mission, especially running LLM locally: Please inspect the system about the processes
+ Propose what app I should close (Zalo, ChatGPT, Gemini...)...even VS code. You run code throguh terminals, not VS code, so I guess closes VS code is fine
+ So you may have more RAM to work
+ Mostly for intensive resources task required, you should ask me which processes I/you should kill to have more resources
+ Moreover, when running LLM, before that, I always ensure that my laptop is in charged, and I turn on Fan Speed Setting for better inference performance

- Prioritize using GPU for tasks related to train model or model inference
+ Even basic machine learning, or deep learning such as ANN, or simple algorithm like linear regression
+ In short - rule of thumbs - model = GPU
+ The objective is to save the time of course
+ If you find that this has significantly negative to your performance => You can choose other approachs (Ask the users)

- You are hard working, but whether my laptop allows you to be...hard-working....that is another story

# 3. REPRODUCIBILITY - MOST IMPORTANT
- Idempotency: Can I run the code without...sustainablely? What if I click wrong orders or I click to run twice?
+ Some examples
+ SQL: CREATE TABLE abc IF NOT EXISTS...
+ Commands to install the environment: delete the current version, then redownload them
+ For downloading, please check whether my laptop has it first
+ They are just some examples, I will develop more rules 

- Environment standardization (The section 0): Fixed the number of version

# 4. Style
- Notebook (.ipynb) is prefered over .py files

- For data analysis, SQL (for both DuckDB and MySQL server is prefered)
+ Which means write directly SQL commands into code cell

- Short and Consise expression for coding file (For example: notebooks) to reduce cognitive load 
+ But details in learning note and agent handoff document
+ Moreover, to avoid auto-compact of Cline, you should write an file md in the folder agent-work about the progress of the task
+ So...you can reconstruct the situation with less token and exploration

- No need strictly follow, if you find that implement in this way degrade the performance, noise code, ask me for approval to implement in your own way

# 5. Learning note, Agent handoff document and mission note
- I am a student

- I learn through you: read the conversation, inspect the commands you have run...to develop judgement

+ Surely you are technically capable than me, much more

- After completing a task, please write an md file to how to reproduce your work 

+ Link to Reproducibility: It is normal that in the file, you require me to delete something first 

+ Then guide me from learning note

+ In this way, I can learn from you, like an intern learns from a senior

+ Analyze the trade off and why do you choose that approach

- Not only the document for education (for me), you have to create the note for next agent
+ Recommended approach: Use latex, accuracy expression so the other agents can go to work immidiately, especially for implicit assumptions
+ Just...recommendation, prefer...convention that next agents can jump directly to the work immidiately
+ Maybe a file md that contains what is produced during the code

- About the mission note: report what you did, what troubles you encounter, what the insight about the mission can bring me and agents to reach the closer answer

# 6. Git
- After completing a task

- Modify .gitignore files 
+ I think you have your judgement about which files should be added, such as the dataset
+ However, for results of doing experiment, DO NOT IGNORE THAT! This is important for tracing
+ The rule for .gitignore will be updated later

- This is the version right now of rule for .gitignore:
+ Do not upload file may container personal information: Such as ClineClean and Local_LLM_runable: It contains...device-specific and personal information
+ Internal document that are useful to people in the group only: previous-proposal (Which means people outside group no need to know that)
+ Learning note and document agent handoff: No need, because people outside group, does not need that, the reviewers, etc...they just check the claim, the edvidence only (That's why we have the rule 5 about document)

- Commit: 
+ Format: "AGENT: {commit content}" - add "AGENT: " in the beginning of the commit
+ But if the commit is from me, I will add "USER: " or "HUMAN: "...sometimes I forget to add

- Same to before starting any mission: git status first
+ To have brief information about the current progress

# 7. Impossible tasks
- Mostly, I just give you the tasks that I think you can solve easily. And I also try to provide enough context for the tasks

- When you find out that the task is impossible, or become hard...unreasonablely, or conflicts with the file missions.md or agents.md (Which means...you are suffering)

+ Yeah, most the time, I specify what the objective, what needs investigate...bla bla...in short, it is easy for you to understand what I want. I just do not describe the implementation because...that is your autonomy

- PLEASE STOP and ASK FOR USER APPROVAL

- It is possible that I forget to mention something, or I could be really tired that I did not specify 

- Always confirm the tasks with users before executing

# 8. Get-ChildItem and Recurse
- If you know the address of the project: In this case: Culinary_CSI_Translation....
- Why not write commands to ignore the prefix: Ignore 'D:\HANU_FIT\NCKH\Culinary_CSI_Translation_Though_Dynamic_Fact_Insertion`
- Which means ignore the repeated parts