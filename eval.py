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

    return content_str


def split_content(content, split_ratio=0.8):
    """
    将content拆分为提示部分和预期续写部分。
    使用句子分割，确保在句子结束处拆分。
    """
    sentences = nltk.sent_tokenize(content, language='english')
    if len(sentences) < 2:
        return content, ""
    
    split_point = math.floor(len(sentences) * split_ratio)
    prompt = ' '.join(sentences[:split_point])
    label = ' '.join(sentences[split_point:])
    return prompt, label

rouge = Rouge()

df = pd.read_parquet('./eval.parquet')

total_bleu = 0.0
total_rouge_f1 = 0.0
count = 0

sample_evaluations = []

for index, row in df.iterrows():
    content = row['content']
    prompt, expected_continuation = split_content(content)
    
    if not expected_continuation.strip():
        continue
    
    generated_continuation = send_message(text_agent, message=prompt)
    
    # 计算BLEU分数
    reference = [nltk.word_tokenize(expected_continuation)]
    hypothesis = nltk.word_tokenize(generated_continuation)
    bleu_score = sentence_bleu(reference, hypothesis, weights=(0.5, 0.5))
    
    # 计算ROUGE分数
    if generated_continuation.strip() and expected_continuation.strip():
        rouge_scores = rouge.get_scores(generated_continuation, expected_continuation)
        rouge_f1 = rouge_scores[0]['rouge-l']['f']
    else:
        rouge_f1 = 0.0
    
    
    total_bleu += bleu_score
    total_rouge_f1 += rouge_f1
    count += 1
    
    if index < 5:
        sample_evaluations.append({
            'title': row['title'],
            'prompt': prompt,
            'expected_continuation': expected_continuation,
            'generated_continuation': generated_continuation,
            'bleu_score': bleu_score,
            'rouge_f1': rouge_f1
        })
    
    print(f"已处理第 {index+1} 行，共 {len(df)} 行")


average_bleu = total_bleu / count if count > 0 else 0
average_rouge_f1 = total_rouge_f1 / count if count > 0 else 0

print("\n=== 评估结果 ===")
print(f"总评估样本数: {count}")
print(f"平均 BLEU 分数: {average_bleu:.4f}")
print(f"平均 ROUGE-L F1 分数: {average_rouge_f1:.4f}")

print("\n=== 部分评估样本 ===")
for sample in sample_evaluations:
    print(f"\n标题: {sample['title']}")
    print(f"提示 (Prompt): {sample['prompt']}")
    print(f"预期续写 (Expected Continuation): {sample['expected_continuation']}")
    print(f"生成续写 (Generated Continuation): {sample['generated_continuation']}")
    print(f"BLEU 分数: {sample['bleu_score']:.4f}")
    print(f"ROUGE-L F1 分数: {sample['rouge_f1']:.4f}")

