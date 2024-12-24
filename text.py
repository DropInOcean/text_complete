import os
os.environ["DASHSCOPE_API_KEY"] = ""
from langchain_community.chat_models.tongyi import ChatTongyi

from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

from dashscope import Assistants, Messages, Runs, Threads
import pandas as pd
import math
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import re

nltk.download('punkt')

llm = ChatTongyi(temperature=0.0)

def create_assistant():
    assistant = Assistants.create(
        model='qwen-turbo',
        name='smart text complete',
        description='文本续写工具',
        instructions='你能够根据用户输入的文本，紧接着续写文本，不要重复用户的输入, 文本无论中文还是英文都可以续写', 
    )

    return assistant

text_agent = create_assistant()


def send_message(assistant, message='描述'):

    thread = Threads.create()

    message = Messages.create(thread.id, content=message)

    response = Runs.create(thread.id, assistant_id=assistant.id, stream=True)
    
    content_str = ""

    for event, run in response:

        if event == "thread.message.delta":

            content_str += run.delta.content.text.value
            
            if len(content_str) >=6:
                content_str = content_str[:6]
                break

    return content_str


def interactive_input():
    history =""
    while True:
        user_input = input("请输入文本 (输入 'exit' 退出): ")
        
        if user_input.lower() == 'exit':
            print("退出程序")
            break
        
        result = send_message(assistant=text_agent, message=user_input)
        print(f"续写结果: {result}")
        
            
interactive_input()

