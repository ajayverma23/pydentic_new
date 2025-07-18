'''
Workflow Structure
Level 1: Content Generation
Multiple agents (each with a different LLM) generate content drafts independently.

Level 2: Consolidation
A consolidator agent merges and summarizes all Level 1 outputs into a single, high-quality article.

Level 3: Enrichment
Image Agent: Creates an image prompt and/or suggests an image URL based on the consolidated content.

Tagline Agent: Generates a tagline, hashtags, alternative titles, and reference data/URLs from the consolidated content.

Level 4: Final Formatting
A final agent takes the consolidated content, image prompt/URL, tagline, hashtags, and alternative titles, and produces the final requested content type, formatting it with bullet points, headings, etc.

Output Storage
Use a dictionary or structured object to store each agent's output for reference and cross-verification.

TBC:

'''



from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI  # Requires langchain-openai package

import sys
import os
import requests
from pathlib import Path
import pandas as pd

# Add the project root to Python's path
#sys.path.append(str(Path(__file__).parent.parent))  # Adjust based on your structure

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from app_schema.data_validation import ContentRequest, ContentResult, ImprovedInputs

from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from langchain_anthropic import ChatAnthropic
from langchain_perplexity import ChatPerplexity  # Hypothetical; replace with actual import if available
from langchain_groq import ChatGroq
from crewai_tools import DallETool
import time
from crewai_tools import (
    FirecrawlScrapeWebsiteTool,
    ScrapeElementFromWebsiteTool,
    YoutubeChannelSearchTool,
    YoutubeVideoSearchTool,
    TXTSearchTool,
    WebsiteSearchTool
)

from datetime import datetime

from dotenv import load_dotenv

# Make sure to use the correct path and forward slashes or raw string
load_dotenv(override=True, dotenv_path="D:/Aj/GenAI/pydentic_new/.env")

from agents.app_pdf_utils import markdown_to_html, html_to_pdf, img_url_to_base64, slugify, \
dedupe_lines, img_file_to_base64, extract_url_from_output, log_status, generate_ppt_from_content_request, log_inputs,\
HARDCODED_PROMPTS, get_final_prompts


class MultiLevelCrewAI:
    def __init__(self):

        self.llms = {}
        self.reload_llms()

    def reload_llms(self):
        self.llms = {
            "GPT": ChatOpenAI(model="gpt-4o", api_key=os.getenv("OPENAI_API_KEY")),
            "Mistral": ChatMistralAI(model="mistral/mistral-large-latest", api_key=os.getenv("MISTRAL_API_KEY")),
            #"GROQ_llama3": ChatGroq(temperature=0, model_name="groq/llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY")), # Limit: 6,000 TPM
            #"GROQ_llama3": ChatGroq(temperature=0, model_name="groq/llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY")),
            #"GROQ_llama3": ChatGroq(temperature=0, model_name="groq/llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"), max_tokens=5000),
            "GROQ_google": ChatGroq(temperature=0, model_name="groq/gemma2-9b-it", api_key=os.getenv("GROQ_API_KEY")),
        }

        # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        # MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
        # GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        #print(f"GPT:{OPENAI_API_KEY}, Mistral:{MISTRAL_API_KEY}, GROQ:{GROQ_API_KEY}")


    def check_llms(self):
        test_prompt = "Hello, are you available?"
        llm_status = {}
        for name, llm in self.llms.items():
            try:
                # Adjust the call according to your LLM class interface
                response = llm.generate(test_prompt) if hasattr(llm, 'generate') else llm(test_prompt)
                llm_status[name] = "working"
            except Exception as e:
                llm_status[name] = f"not working: {e}"
        return llm_status
    


    def run(self, content_request):

        llm_status = self.check_llms()
        print("llm_status:", llm_status)

        short_title = slugify(content_request.subject or "output")
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_filename = f"{short_title}_{timestamp}"

        images_dir = "D:\Aj\GenAI\pydentic_new\output\images"
        logs_dir = "D:\Aj\GenAI\pydentic_new\output\logs"
        content_dir = "D:\Aj\GenAI\pydentic_new\output\content"

        image_path = os.path.join(images_dir, f"{base_filename}_Image.png")
        log_path = os.path.join(logs_dir, f"{base_filename}_logfile.csv")
        csv_path = os.path.join(logs_dir, f"{base_filename}_Dashboard.csv")
        output_pdf_path = os.path.join(content_dir, f"{base_filename}_Content.pdf")
        output_md_path = os.path.join(content_dir, f"{base_filename}_Content.md")
        output_html_path = os.path.join(content_dir, f"{base_filename}_Content.html")
        output_txt_path = os.path.join(content_dir, f"{base_filename}_Metadata.txt")
        output_ppt_path = os.path.join(content_dir, f"{base_filename}_Content.pptx")
        output_userinput_path = os.path.join(content_dir, f"{base_filename}_userinput.csv")
        
        agent_outputs = {}
        agent_time = {}

        # Level 0: Pre-Level 1: User Input Review Agent
        level0_agents = []
        level0_tasks = []

        system_prompt, user_prompt = get_final_prompts(content_request, HARDCODED_PROMPTS)

        review_agent = Agent(
            role="Input Review Agent",
            goal="Review and improve user inputs for content generation.",
            backstory="Expert at clarifying and enhancing content briefs.",
            llm=self.llms["GPT"]
            #llm=self.llms["GROQ_llama3"]
        )

        review_task = Task(
            description=(
                "Review the following user inputs for completeness and clarity. "
                "Improve or fill in missing details for: user_input, reference_data, context, subject, topic, system_prompt, user_prompt "
                "Additionally, search for and add the top 2-3 relevant and authoritative webpages and YouTube videos related to the content topic, "
                "and include their URLs in the reference_data field. "
                "Modify or expand the user_input if needed to ensure the content request is clear, detailed, and well-structured for optimal content creation. "
                "Return the improved values as a JSON object with the same keys: user_input, reference_data, context, subject, topic, system_prompt, user_prompt"
                f"\n\nUser Input: {content_request.user_input}"
                f"\nReference Data: {content_request.reference_data}"
                f"\nContext: {content_request.context}"
                f"\nSubject: {content_request.subject}"
                f"\nTopic: {content_request.topic}"
                f"\nSystem Prompt: {system_prompt}"
                f"\nUser Prompt: {user_prompt}"
                #f"\nSystem Prompt: {content_request.system_prompt}"
                #f"\nUser Prompt: {content_request.user_prompt}"
            ),
            agent=review_agent,
            expected_output=(
                "Improved user inputs as a JSON object with keys: "
                "user_input, reference_data (including new URLs), context, subject, topic, system_prompt, user_prompt"
            ),
            output_pydantic=ImprovedInputs
        )


        level0_agents.append(review_agent)
        level0_tasks.append(review_task)

        crew0 = Crew(agents=level0_agents, tasks=level0_tasks, process=Process.sequential)
        start_time0 = time.time()
        level0_results = crew0.kickoff()
        end_time0 = time.time()
        agent_time['level0'] = round(end_time0 - start_time0, 2)
        agent_outputs['level0'] = level0_results

        improved_inputs = level0_results.pydantic.dict()
        #agent_outputs['level0'] = improved_inputs

        # Log original and improved user inputs
        log_inputs(content_request, improved_inputs, log_path=output_userinput_path)

        for i, (agent, task) in enumerate(zip(level0_agents, level0_tasks)):
            try:
                # Safely get output content
                output_content = None
                if (
                    hasattr(level0_results, 'tasks_output') 
                    and i < len(level0_results.tasks_output)
                    and hasattr(level0_results.tasks_output[i], 'raw')
                ):
                    output_content = level0_results.tasks_output[i].raw  # or .result
                status = "completed" if output_content else "failed"
                log_status(log_path, level=0, agent=agent, task=task, status=status, output=output_content)  # Pass string, not TaskOutput
            except Exception as e:
                log_status(log_path, level=0, agent=agent, task=task, status="error", error=str(e))


        # Level 1: Content Generation
        content_agents = []
        content_tasks = []
        #system_prompt, user_prompt = get_final_prompts(content_request, HARDCODED_PROMPTS)

        # Create a directory to store llm outputs

        for name, llm in self.llms.items():
            agent = Agent(
                #role=f"{name} Content Agent",
                #role=content_request.role,
                role=f"{content_request.role} (LLM: {name})",
                goal=(
                    f"Generate {content_request.content_type} for {content_request.target_audience} "
                    f"at {content_request.audience_level} level."
                ),
                #backstory=f"{name} is an expert content creator.",
                #backstory=system_prompt,
                backstory=improved_inputs["system_prompt"],
                llm=llm
            )
            content_agents.append(agent)
            content_tasks.append(Task(
                description=(

                    # improved_inputs will have user_input, reference_data, context, subject, topic, 
                    #f"{user_prompt}\n"
                    f"{improved_inputs['user_prompt']}\n"
                    f"Create a {content_request.content_type} as per the following details:\n"
                    f"- User Input: {improved_inputs['user_input']}\n"
                    f"- Reference Data: {improved_inputs['reference_data']}\n"
                    f"- Context: {improved_inputs['context']}\n"
                    f"- Subject: {improved_inputs['subject']}\n"
                    f"- Topic: {improved_inputs['topic']}\n"
                    f"- Domain: {content_request.domain}\n"
                    f"- Target Audience: {content_request.target_audience}\n"
                    f"- Audience Level: {content_request.audience_level}\n"
                    f"- Content Size: {content_request.content_size} words\n"
                    f"Ensure the content is tailored for the specified domain and audience."
                    f"Generate a **minimum 100,000-word** document with:\n"
                        "- 10+ detailed sections\n"
                        "- 5+ case studies per section\n"
                        "- 20+ references\n"
                        "- Tables and code examples"
                ),
                agent=agent,
                expected_output=f"{content_request.content_type} draft."
            ))

        level1_agents = content_agents # for same naming convention
        level1_tasks = content_tasks

        # After running llm tasks, save outputs
        for task in content_tasks:
            llm_name = task.agent.role.split("LLM:")[-1].strip()  # Extract LLM name from role     
            output_llm_pdf_path = os.path.join(content_dir,f"{base_filename}_Content_{llm_name}.pdf")
            output_llm_html_path = os.path.join(content_dir, f"{base_filename}_Content_{llm_name}.html")
                                  
            if task.output:
                # Convert to HTML
                html_text = markdown_to_html(task.output)

                # Save as HTML
                with open(output_llm_html_path, "w", encoding="utf-8") as f:
                    f.write(html_text)

                # Generate PDF from HTML
                html_to_pdf(html_text, output_llm_pdf_path)                         
                                               
                                           
        
        crew1 = Crew(agents=content_agents, tasks=content_tasks, process=Process.sequential)
        start_time1 = time.time()
        level1_results = crew1.kickoff()
        end_time1 = time.time()
        agent_time['level1'] = round(end_time1 - start_time1, 2)
        agent_outputs['level1'] = level1_results

        for i, (agent, task) in enumerate(zip(level1_agents, level1_tasks)):
            try:
                # Safely get output content
                output_content = None
                if (
                    hasattr(level1_results, 'tasks_output') 
                    and i < len(level1_results.tasks_output)
                    and hasattr(level1_results.tasks_output[i], 'raw')
                ):
                    output_content = level1_results.tasks_output[i].raw  # or .result
                status = "completed" if output_content else "failed"
                log_status(log_path, level=1, agent=agent, task=task, status=status, output=output_content)  # Pass string, not TaskOutput
            except Exception as e:
                log_status(log_path, level=1, agent=agent, task=task, status="error", error=str(e))
                

        # Level 2: Consolidation
        # Dynamically build llm_outputs dictionary
        llm_outputs = {}
        for task in content_tasks:
            # Extract LLM name from the agent's role (assuming your naming convention is consistent)
            llm_name = task.agent.role.split("LLM:")[-1].strip()
            llm_outputs[llm_name] = task.output

        # for k, v in llm_outputs.items():
        #     print(f"KEY: {k}")
        #     print(f"TYPE: {type(v)}")
        #     print(f"DIR: {dir(v)}")
        #     print(f"STR: {str(v)}")
        #     print("===")

        # Find the LLM with the largest output (by word count)
        def get_output_text(obj):
            return str(obj) if obj else ""

        largest_llm_name = max(llm_outputs, key=lambda k: len(get_output_text(llm_outputs[k]).split()))
        largest_output = llm_outputs[largest_llm_name]

        level2_agents = []
        level2_tasks = []

        consolidate_agent = Agent(
            role="Consolidator",
            goal="Combine, deduplicate, and expand all Level 1 LLM-generated drafts into a detailed, comprehensive document matching or exceeding the total input length.",
            backstory="Expert at merging content, preserving all unique information, and expanding brief points to create in-depth, cohesive documents.",
            max_tokens=32000,
            llm=self.llms["GPT"]
            #llm=self.llms["GROQ_llama3"]
        )

        llm_full_texts = []
        for name, output in llm_outputs.items():
            if name == largest_llm_name:
                llm_full_texts.append(f"Base content ({name}):\n{output}\n")
            else:
                llm_full_texts.append(f"{name} content:\n{output}\n")

        all_llm_content = "\n".join(llm_full_texts)


        consolidate_description = (
            #f"Use the {largest_llm_name} output (largest at {len(largest_output.split())} words) as the base. "
            f"Use the {largest_llm_name} output (largest at {len(str(largest_output).split())} words) as the base. "
            "Compare with all other LLM outputs to:\n"
            "1. Identify unique content not in the base\n"
            "2. Add unique content to the base\n"
            "3. Ensure final output is LONGER than the base\n"
            "4. Never remove content, only add\n\n"
            f"{all_llm_content}"
        )

        # Consolidation task
        consolidate_task = Task(
            description = consolidate_description,
            agent=consolidate_agent,
            expected_output="Expanded consolidated draft longer than base content.",
            context=content_tasks
        )

        level2_agents.append(consolidate_agent)
        level2_tasks.append(consolidate_task)

        crew2 = Crew(agents=level2_agents, tasks=level2_tasks, process=Process.sequential)
        start_time2 = time.time()
        level2_results = crew2.kickoff()
        end_time2 = time.time()
        agent_time['level2'] = round(end_time2 - start_time2, 2)
        agent_outputs['level2'] = level2_results

        for i, (agent, task) in enumerate(zip(level2_agents, level2_tasks)):
            try:
                # Safely get output content
                output_content = None
                if (
                    hasattr(level2_results, 'tasks_output') 
                    and i < len(level2_results.tasks_output)
                    and hasattr(level2_results.tasks_output[i], 'raw')
                ):
                    output_content = level2_results.tasks_output[i].raw  # or .result
                status = "completed" if output_content else "failed"
                log_status(log_path, level=2, agent=agent, task=task, status=status, output=output_content)  # Pass string, not TaskOutput
            except Exception as e:
                log_status(log_path, level=2, agent=agent, task=task, status="error", error=str(e))

        # Level 3
        level3_agents = []
        level3_tasks = []

        image_agent = Agent(
            role="Image Creator",
            goal="Generate an image for the article using DALL-E.",
            backstory="Visual expert.",
            #llm=ChatOpenAI(model="gpt-4-turbo"),  # or "gpt-4o"
            llm=self.llms["GPT"],
            #llm=self.llms["GROQ_llama3"],
            tools=[DallETool()]  # <-- This enables real image generation
        )

        image_task = Task(
            description="Generate an image prompt or suggest an image URL for the consolidated article.",
            agent=image_agent,
            #expected_output="Image prompt/URL.",
            expected_output="Image URL.",
            context = [consolidate_task]
            )
        
        level3_agents.append(image_agent)
        level3_tasks.append(image_task)

        tag_agent = Agent(
            role="Tagline & Hashtag Generator",
            goal="Create tagline, hashtags, titles, and reference URLs.",
            backstory="Marketing copy expert.",
            #llm=self.llms["GROQ_llama3"]
            llm=self.llms["GPT"]
        )
        
        tag_task = Task(
            description=(
                "Generate the following for the article:\n"
                "- Tagline\n"
                "- Hashtags (comma separated)\n"
                "- Alternative Titles (at least two)\n"
                "- Reference (if any)\n\n"
                "Format your response in Markdown as:\n"
                "Tagline: <your tagline>\n"
                "Hashtags: <your hashtags>\n"
                "Alternative Titles: <your alternative titles>\n"
                "Reference: <your reference>\n\n"
                "Example:\n"
                "Tagline: Unlocking AI for Everyone\n"
                "Hashtags: #AI, #GenAI, #Blog\n"
                "Alternative Titles: Demystifying GenAI; GenAI for Beginners\n"
                "Reference: www.xyz.com"
            ),
            agent=tag_agent,
            expected_output=(
                "A Markdown block with:\n"
                "Tagline: <your tagline>\n"
                "Hashtags: <your hashtags>\n"
                "Alternative Titles: <your alternative titles>\n"
                "Reference: <your reference>"
            ),
            context = [consolidate_task]
        )

        level3_agents.append(tag_agent)
        level3_tasks.append(tag_task)

        # Wikipedia Agent
        wiki_agent = Agent(
            role="Wikipedia Search Agent",
            goal=f"Search Wikipedia for '{content_request.user_input}' and extract the most relevant summary and facts.",
            backstory="Expert at extracting precise information from Wikipedia.",
            #llm=self.llms["GROQ_llama3"],
            llm=self.llms["GPT"],
            tools=[WebsiteSearchTool()]
        )
        wiki_task = Task(
            description=f"Search Wikipedia for '{content_request.user_input}'. Extract and summarize the most relevant article sections, definitions, and facts.",
            agent=wiki_agent,
            expected_output="Wikipedia summary and key facts."
        )

        level3_agents.append(wiki_agent)
        level3_tasks.append(wiki_task)

        # Research Paper Agent
        research_agent = Agent(
            role="Research Paper Search Agent",
            goal=f"Find research papers related to '{content_request.user_input}' and summarize their main findings.",
            backstory="Skilled in academic search and summarization.",
            #llm=self.llms["GROQ_llama3"],
            llm=self.llms["GPT"],
            tools=[TXTSearchTool()]
        )
        research_task = Task(
            description=f"Search academic databases for research papers about '{content_request.user_input}'. Provide abstracts and summarize key findings.",
            agent=research_agent,
            expected_output="Research paper summaries and key findings."
        )

        level3_agents.append(research_agent)
        level3_tasks.append(research_task)

        # YouTube Agent (only if user provides a YouTube link; otherwise, can skip or search by keyword)
        youtube_agent = Agent(
            role="YouTube Video Summary Agent",
            goal=f"Summarize the content of the YouTube video related to '{content_request.reference_data}' if a link is provided.",
            backstory="Expert at analyzing and summarizing YouTube content.",
            #llm=self.llms["GROQ_llama3"],
            llm=self.llms["GPT"],
            tools=[YoutubeVideoSearchTool()]
        )
        youtube_task = Task(
            description=f"If a YouTube link is provided, extract and summarize the main points of the video related to '{content_request.reference_data}'.",
            agent=youtube_agent,
            expected_output="YouTube video summary."
        )

        level3_agents.append(youtube_agent)
        level3_tasks.append(youtube_task)

        # Webpage Scraper Agent (only if a webpage link is provided)
        webpage_agent = Agent(
            role="Webpage Scraping Agent",
            goal=f"Scrape and summarize the content of the provided webpage relevant to '{content_request.reference_data}'.",
            backstory="Expert at web content extraction.",
            #llm=self.llms["GROQ_llama3"],
            llm=self.llms["GPT"],
            tools=[ScrapeElementFromWebsiteTool()]
        )

        # In Level 3 agent setup
        webpage_task = Task(
            description=(
                f"Scrape and summarize the webpage content ONLY IF a valid webpage URL (not YouTube) is provided. "
                f"Current input: '{content_request.reference_data}'. "
                "Skip if no webpage link exists."
            ),
            agent=webpage_agent,
            expected_output="Webpage summary or 'No webpage link provided'."
        )


        level3_agents.append(webpage_agent)
        level3_tasks.append(webpage_task)

        crew3 = Crew(agents=level3_agents, tasks=level3_tasks, process=Process.sequential)
        start_time3 = time.time()
        level3_results = crew3.kickoff()
        end_time3 = time.time()
        agent_time['level3'] = round(end_time3 - start_time3, 2)
        agent_outputs['level3'] = level3_results

        #print("Agent raw tag output:", agent_outputs['level3'].tasks_output[1])

        for i, (agent, task) in enumerate(zip(level3_agents, level3_tasks)):
            try:
                # Safely get output content
                output_content = None
                if (
                    hasattr(level3_results, 'tasks_output') 
                    and i < len(level3_results.tasks_output)
                    and hasattr(level3_results.tasks_output[i], 'raw')
                ):
                    output_content = level3_results.tasks_output[i].raw  # or .result
                status = "completed" if output_content else "failed"
                log_status(log_path, level=3, agent=agent, task=task, status=status, output=output_content)  # Pass string, not TaskOutput
            except Exception as e:
                log_status(log_path, level=3, agent=agent, task=task, status="error", error=str(e))


        # Level 4: Final Formatting
        level4_agents = []
        level4_tasks = []

        final_agent = Agent(
            role="Content Consolidator & Formatter",
            goal="Merge and format all outputs into 3-4 pages of concise, non-redundant content",
            backstory="Expert technical writer skilled at synthesizing complex information",
            llm=self.llms["GPT"],
            #llm=self.llms["GROQ_llama3"],
            memory=True  # Essential for tracking cross-task context
        )

        final_task = Task(
            description=(
                "Format consolidated content into 3-4 professional pages. Requirements:\n"
                "1. Inputs: Consolidated draft + research/wiki/web/video data\n"
                "2. Add tags, tables, and visual elements\n"
                "3. ENSURE OUTPUT IS ≥ 4000 WORDS\n"
                "4. If too short, expand with examples/case studies\n\n"
                "Conditional Check:\n"
                "IF word_count(consolidated_output) < word_count(largest_input):\n"
                "   THROW ERROR: 'Content too short - regenerate with expansion'"
            ),
            agent=final_agent,
            expected_output=(
                "A final, fully formatted, and expanded document (minimum 3-4 pages, at least 4000 words) containing all required sections."
            ),
            #expected_output="Final 3-4 page document with ≥ 2000 words",
            #context=[consolidate_task, research_task, wiki_task, youtube_task, webpage_task]
            context=[consolidate_task, image_task, tag_task, wiki_task, research_task, youtube_task, webpage_task],
            output_file="final_report.md"
        )

        level4_agents.append(final_agent)
        level4_tasks.append(final_task)

        crew4 = Crew(agents=[final_agent], tasks=[final_task], process=Process.sequential)
        start_time4 = time.time()
        #level4_results = crew4.kickoff()
        final_result = crew4.kickoff()
        end_time4 = time.time()
        agent_time['level4'] = round(end_time4 - start_time4, 2)
        agent_outputs['final'] = final_result

        if 'final' not in agent_outputs:
            log_status(log_path, level=4, agent='Final Formatter', task='Final Output', status='error', error='Final output missing')

        for i, (agent, task) in enumerate(zip(level4_agents, level4_tasks)):
            try:
                # Safely get output content
                output_content = None
                if (
                    hasattr(final_result, 'tasks_output') 
                    and i < len(final_result.tasks_output)
                    and hasattr(final_result.tasks_output[i], 'raw')
                ):
                    output_content = final_result.tasks_output[i].raw  # or .result
                status = "completed" if output_content else "failed"
                log_status(log_path, level=4, agent=agent, task=task, status=status, output=output_content)  # Pass string, not TaskOutput
            except Exception as e:
                log_status(log_path, level=4, agent=agent, task=task, status="error", error=str(e))

        # Level 5: Final Formatting
        level5_agents = []
        level5_tasks = []

        qa_agent = Agent(
            role="QA Agent",
            goal="Assess the quality, confidence, errors, and timing of all previous agent outputs.",
            backstory="An expert evaluator for AI-generated content and process quality.",
            llm=self.llms["GPT"]
            #llm=self.llms["GROQ_llama3"]
            #llm=self.llms.get('qa') or list(self.llms.values())[0],  # Pick a suitable LLM
        )

        # Prepare a summary of all outputs for the QA agent to review
        all_outputs_summary = ""
        for level, (agents, tasks, outputs) in enumerate([
            (level1_agents, level1_tasks, agent_outputs.get('level1')),
            (level2_agents, level2_tasks, agent_outputs.get('level2')),
            (level3_agents, level3_tasks, agent_outputs.get('level3')),
            (level4_agents, level4_tasks, agent_outputs.get('final')),
        ], start=1):
            for i, (agent, task) in enumerate(zip(agents, tasks)):
                output = None
                if outputs and hasattr(outputs, 'tasks_output') and i < len(outputs.tasks_output):
                    output = outputs.tasks_output[i]
                content_str = getattr(output, "raw", None) if output else None
                all_outputs_summary += (
                    f"\nLevel {level} - Agent: {agent.role}\n"
                    f"Task: {task.description}\n"
                    f"Output: {content_str[:500] if content_str else 'No output'}\n"
                )

        qa_task = Task(
            description=(
                "You are to review the following agent outputs from all levels of the content generation process. "
                "For each, provide:\n"
                "- Quality score (1-10)\n"
                "- Confidence score (1-10)\n"
                "- Any errors or issues detected\n"
                "- Suggestions for improvement\n\n"
                "Format your response as a JSON list of objects with keys: level (integer), agent (string), quality (int), confidence (int), errors (string), suggestions (string).\n"
                f"Here are the outputs:\n{all_outputs_summary}"
            ),
            agent=qa_agent,
            expected_output="A JSON list of QA evaluations for each agent output."
        )



        level5_agents.append(qa_agent)
        level5_tasks.append(qa_task)

        crew5 = Crew(agents=level5_agents, tasks=level5_tasks, process=Process.sequential)
        start_time5 = time.time()
        level5_results = crew5.kickoff()
        end_time5 = time.time()
        agent_time['level5'] = round(end_time5 - start_time5, 2)
        agent_outputs['level5'] = level5_results

        for i, (agent, task) in enumerate(zip(level5_agents, level5_tasks)):
            try:
                # Safely get output content
                output_content = None
                if (
                    hasattr(level5_results, 'tasks_output') 
                    and i < len(level5_results.tasks_output)
                    and hasattr(level5_results.tasks_output[i], 'raw')
                ):
                    output_content = level5_results.tasks_output[i].raw  # or .result
                    #print(f'level 5 output_content:', output_content) #####
                status = "completed" if output_content else "failed"
                log_status(log_path, level=5, agent=agent, task=task, status=status, output=output_content)  # Pass string, not TaskOutput
            except Exception as e:
                log_status(log_path, level=5, agent=agent, task=task, status="error", error=str(e))


        #dashboard
        dashboard_records = []

        for level, (agents, tasks, outputs) in enumerate([
            (level0_agents, level0_tasks, agent_outputs.get('level0')),
            (level1_agents, level1_tasks, agent_outputs.get('level1')),
            (level2_agents, level2_tasks, agent_outputs.get('level2')),
            (level3_agents, level3_tasks, agent_outputs.get('level3')),
            (level4_agents, level4_tasks, agent_outputs.get('final')),
            (level5_agents, level5_tasks, agent_outputs.get('level5')),
        ], start=1):
            
            for i, (agent, task) in enumerate(zip(agents, tasks)):
                output = None
                # 1. Check if outputs and tasks_output exist
                if outputs and hasattr(outputs, 'tasks_output') and i < len(outputs.tasks_output):
                    output = outputs.tasks_output[i]
                else:
                    # Optionally log that this output is missing
                    log_status(log_path, level, agent, task, "error", output=None, error="No output for this task (index out of range)")

                # 2. Extract content from .raw (not .content)
                content_str = getattr(output, "raw", None) if output else None  # KEY CHANGE
                #content_str = getattr(output, "content", None) if output else None
                
                content_length = len(content_str) if content_str else None
                #content_length = len(output) if output else None
                status = "Executed" if content_str else "Not Executed"
                # dashboard_records.append({
                #     "Level": level,
                #     "Agent": agent.role,
                #     "Task": task.description,
                #     "Status": status,
                #     "Content Length": content_length,
                #     #"Output Preview": output[:100] if output else None,
                #     "Output Preview": content_str[:100] if content_str else None,
                #     # Add quality/confidence/error/time as available
                # })

                dashboard_records.append({
                    "Level": level,
                    "Agent": agent.role,
                    "Task": task.description,
                    "Status": status,
                    "Content Length": content_length,
                    "Output Preview": content_str[:100] if content_str else None,
                    "Quality": None,  # Initialize keys
                    "Confidence": None,
                    "Errors": None,
                    "Suggestions": None
                })


        import re
        import json

        qa_feedback = []

        qa_raw = getattr(level5_results, "raw", "[]")
        # Remove Markdown code block if present
        # Remove opening and closing Markdown code block markers (```json or `````` at end)
        qa_raw_clean = re.sub(r"^```(?:json)?\n?|```$", "", qa_raw.strip(), flags=re.MULTILINE).strip()  # gemini

        try:
            qa_feedback = json.loads(qa_raw_clean)
        except Exception as e:
            print("Failed to parse QA agent output:", e)
            print("Raw QA output after cleaning:", qa_raw_clean)

        #print("Dashboard records before QA mapping:", len(dashboard_records))
        #print("QA feedback entries:", len(qa_feedback))


        # for feedback in qa_feedback:
        #     #print("QA feedback agent/level:", feedback.get("agent"), feedback.get("level"))
        #     for record in dashboard_records:
        #         #print("Dashboard record agent/level:", record["Agent"], record["Level"])
        #         # Convert level to integer if necessary
        #         qa_level = feedback.get("level")
        #         if isinstance(qa_level, str) and qa_level.startswith("Level "):
        #             qa_level = int(qa_level.replace("Level ", "").strip())
                
        #         # Check level and agent name match
        #         if (record["Level"] == qa_level and 
        #             feedback.get("agent", "").strip().lower() in record["Agent"].lower()):
                    
        #             # Update dashboard record with QA data
        #             record["Quality"] = feedback.get("quality")
        #             record["Confidence"] = feedback.get("confidence")
        #             record["Errors"] = feedback.get("errors")
        #             record["Suggestions"] = feedback.get("suggestions")

        for feedback in qa_feedback:
            qa_level = feedback.get("level")
            qa_agent = feedback.get("agent", "").strip().replace(" ", "")  # Normalize whitespace
            
            for record in dashboard_records:
                # Convert QA level to match dashboard: Level 1 → Level 2
                dashboard_level = qa_level + 1 if isinstance(qa_level, int) else qa_level
                
                # Normalize record agent name
                record_agent = record["Agent"].replace(" ", "")
                
                if (record["Level"] == dashboard_level and qa_agent in record_agent):  # Substring match
                    print("Match found! Updating record.")
                    record.update({
                        "Quality": feedback.get("quality"),
                        "Confidence": feedback.get("confidence"),
                        "Errors": feedback.get("errors"),
                        "Suggestions": feedback.get("suggestions")
                    })

                    print(f"Checking: QA Level={qa_level} → Dashboard Level={dashboard_level}")
                    print(f"QA Agent='{qa_agent}' vs Record Agent='{record_agent}'")




        # Save all outputs for reference/cross-verification
        final_output = agent_outputs.get('final')
        if final_output is None:
            # Handle the missing output gracefully
            print("Final output not available. Check previous steps for errors.")

        # After running the agent/task
        image_output = image_task.output if hasattr(image_task, "output") else image_task.raw

        print("image_output:", image_output, type(image_output))
        #image_url = extract_url_from_output(image_output)
        tag_output = agent_outputs['level3'].tasks_output[1].raw    # adjust index if needed
        print("tag_output content:", tag_output)
        print("tag_output type:", type(tag_output))

        # Assume image_output is a URL, tag_output is a text block with tagline, hashtags, titles

        # Embed image as base64
        img_tag = ""

        # Ensure you get the raw string from the TaskOutput object
        if hasattr(image_output, "raw") and image_output.raw:
            image_output_str = image_output.raw
        elif hasattr(image_output, "output") and image_output.output:
            image_output_str = image_output.output
        else:
            image_output_str = str(image_output).strip() 
            #image_output_str = str(image_output)  # fallback

        if image_output_str and image_output_str.startswith("http"):
            img_base64 = img_url_to_base64(image_output_str)
            img_tag = f"![Generated Image]({img_base64})\n"
        else:
            img_tag = ""
            print("No valid image URL found in image_output:", image_output_str)

        # image_output is a TaskOutput object
        #image_output_str = getattr(image_output, "raw", None) or getattr(image_output, "output", None) or str(image_output)
        print("Extracted image_output_str:", image_output_str)  # Should be the URL

        image_url = extract_url_from_output(image_output_str)
        print("Image URL for download:", image_url)


        # Compose the markdown
        md_text = ""
        if hasattr(final_output, 'tasks_output'):
            md_text = final_output.tasks_output[0].raw
        elif isinstance(final_output, str):
            md_text = final_output
        else:
            md_text = str(final_output)

        # Append image and tags & signature at the end
        signature_block = """
        ---
        **For detailed insights, please visit my blog at:**  
        https://ajayverma23.blogspot.com/  
        **Explore more of my articles on Medium at:**  
        https://medium.com/@ajayverma23  
        **Connect with me:**  
        https://www.linkedin.com/in/ajay-verma-1982b97/
        """

        # save dashbaord
        df = pd.DataFrame(dashboard_records)  # dashboard_records is your table data
        #csv_path = f"{base_filename}.csv"
        df.to_csv(csv_path, index=False)

        # Download and save the image
        if image_url:
            response = requests.get(image_url)
            if response.status_code == 200:
                with open(image_path, "wb") as f:
                    f.write(response.content)
                print(f"Image saved as {image_path}")
            else:
                print("Image could not be downloaded.")
        else:
            print("No image URL found in agent output.")

        # final Markdown/HTML for PDF generation
        if os.path.isfile(image_path):
            print("Image file exists.")
            img_base64 = img_file_to_base64(image_path)
            img_tag = f"![Generated Image]({img_base64})\n"
        else:
            print("Image file does not exist.")
            img_tag = ""  # Or use a default image / img_tag to your Markdown before converting to HTML/PDF

        print("Image URL:", image_url)
        print("Image path:", image_path)
        print("File exists:", os.path.isfile(image_path))
        print("File size:", os.path.getsize(image_path) if os.path.isfile(image_path) else 0)

        # Format tag_output (ensure it's a string with all needed fields)
        tag_output = dedupe_lines(tag_output)
        print("tag_output:", tag_output, type(tag_output))

        tag_str = ""
        if isinstance(tag_output, dict):
            tag_str = (
                f"**Tagline:** {tag_output.get('tagline', '')}\n"
                f"**Hashtags:** {tag_output.get('hashtags', '')}\n"
                f"**Alternative Title:** {tag_output.get('alternative_title', '')}\n"
                f"**Reference:** {tag_output.get('reference', '')}\n"
            )
        else:
            tag_str = str(tag_output)

        # Now assemble the final Markdown
        full_md = (
            md_text
            + "\n\n" + img_tag
            + "\n\n" + tag_str
            + "\n\n" + signature_block
        )
        

        # Save as Markdown
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(full_md)
        print(f"Markdown saved to {output_md_path}")

        # Convert to HTML
        html_text = markdown_to_html(full_md)

        # Save as HTML
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_text)
        print(f"HTML saved to {output_html_path}")

        # Generate PDF from HTML
        html_to_pdf(html_text, output_pdf_path)
        print(f"PDF saved to {output_pdf_path}")

        # save as pdf
        generate_ppt_from_content_request(content_request, output_ppt_path)

        # save tags/titles separately
        #with open("final_metadata.txt", "w", encoding="utf-8") as f:
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write(tag_output)

        return agent_outputs
    
    
