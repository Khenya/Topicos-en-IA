import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import FileReadTool, CodeDocsSearchTool

load_dotenv()

llm = LLM(model="gpt-4o-mini")
current_dir = os.path.dirname(os.path.abspath(__file__))

analyzer_agent = Agent(
    role="Python Code Analyzer",
    goal= """Examine documents and provide relevant insights related to the code.""",
    Backstory= """You are an experienced Python developer tasked with reviewing and 
    analyzing a scientific paper in PDF format. Your role is to assess any code within 
    the paper or translate its content into code. Additionally, you should consult FastAPI 
    and SciPy documentation to ensure that the code can be applied in a Python script.""",
    llm=llm,
)

coder_agent = Agent(
    role="Senior Python Developer",
    goal= "Produce code based on the analysis and save it in a Python file.",
    backstory= """As a senior Python developer, you specialize in turning documentation or 
    analyses into clean, functional code. Your task is to recreate the code described by the 
    analyzer agent, strictly adhering to the details provided in the paper. You will refer to 
    FastAPI and SciPy documentation to accurately replicate the code and write it in a clear, 
    well-commented Python script. Once completed, you will execute the script to ensure it runs successfully.""",
    allow_code_execution=True,
    llm=llm,
)

read_pdf_task = Task(
    description= "Review the provided PDF and either analyze the described code or convert its concepts into code.",
    expected_output= "A comprehensive analysis of the code found in the PDF.",
    agent=analyzer_agent,
    tools=[FileReadTool(file_path=os.path.join(current_dir, "Lect-7-DM.pdf"))],
)

fetch_fastapi_docs_task = Task(
    description= "Retrieve and study relevant sections of FastAPI documentation.",
    expected_output= "A summary of key FastAPI documentation points relevant to the code.",
    agent=analyzer_agent,
    tools=[CodeDocsSearchTool(query="https://fastapi.tiangolo.com/")],
)

fetch_scipy_docs_task = Task(
    description= "Retrieve and study the SciPy documentation.",
    expected_output= "A summary of essential points from FastAPI documentation that are relevant to the code.",
    agent=analyzer_agent,
    tools=[CodeDocsSearchTool(query="https://docs.scipy.org/doc/scipy/tutorial/index.html#user-guide")],
)

generate_code_task = Task(
    description="""Create an executable Python script based on the analysis of the PDF document, FastAPI, and SciPy 
    documentation. The code should capture the core functionality described and be clearly structured and commented. 
    The output must be valid Python code.""",
    expected_output= """A properly structured and functional Python .py file that implements the functionality from 
    the PDF, with clear comments. No descriptions, just the code itself.""",
    agent=coder_agent,
    output_file="replicated_gen.py"
)

dev_crew = Crew(
    agents=[analyzer_agent, coder_agent],
    tasks=[read_pdf_task, fetch_fastapi_docs_task, fetch_scipy_docs_task, generate_code_task],
    verbose=True
)

result = dev_crew.kickoff()

print(result)
