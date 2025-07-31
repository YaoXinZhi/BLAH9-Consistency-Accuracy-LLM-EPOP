# -*- coding:utf-8 -*-
# ! usr/bin/env python3
"""
Created on 08/01/2024 20:04
@Author: yao
"""


import json
import os.path

from openai import OpenAI
import json

from datetime import datetime
import time

import argparse


def read_doc(sent_file: str):
    with open(sent_file) as f:
        sent = f.read().strip()
    return sent


def get_para():
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', dest='model', choices=['kimi', 'deepseek', 'gpt-4o-mini', 'qwen3'],
                        required=True)

    parser.add_argument('-t', dest='text_file',
                        default='/Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/documents/Texte2.txt')
    parser.add_argument('-p', dest='prompt_file',
                        default='/Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/instructions/short.txt')
    parser.add_argument('-s', dest='save_path',
                        default='/Users/yao/Nutstore Files/Mac2PC/LLM-EPOP-RE_BLAH9_2025-01-15/zero-shot-evaluation-main/experiments/output/repetition/kimi')

    parser.add_argument('-r', dest='temperature', type=float, default=0.2)
    parser.add_argument('-o', dest='top_p', type=float, default=0.1)

    parser.add_argument('-e', dest='repeat_num', type=int, default=5)

    args = parser.parse_args()

    return args


def main():

    # Step-1 data path.
    args = get_para()

    sent_file = args.text_file
    prompt_file = args.prompt_file
    save_path = args.save_path

    # fixme: change the save folder
    if args.temperature != 0.2 or args.top_p != 0.1:
        save_dir = os.path.basename(prompt_file).split('.')[ 0 ] + f'_temperature{args.temperature}_top-p{args.top_p}'
    else:
    # save_dir = os.path.basename(prompt_file).split('.')[ 0 ] + f'_repeat{args.repeat_num}'
        save_dir = os.path.basename(prompt_file).split('.')[ 0 ]
    #     save_dir = '.'.join(os.path.basename(prompt_file).split('.')[:-1])
    # todo 补全欠缺的实验
    #     save_dir = 'TAEC_scenario-v3'

    text_prefix = os.path.basename(sent_file).split('.')[ 0 ]
    repeat_time = args.repeat_num

    dir_path = f'{save_path}/{save_dir}/{text_prefix}'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    # Step-0 api
    if args.model == 'gpt-4o-mini':
        api_key = ''

        model = "gpt-4o-mini"

        client = OpenAI(
            api_key=api_key,
        )
    elif args.model == 'deepseek':
        # xinzhi 25-03-8
        api_key = ''
        print('-------Jingbo Key.-------')

        # DeepSeek-v3
        # model = "deepseek-reasoner"
        model = "deepseek-chat"
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    elif args.model == 'kimi':
        # xzyao-BLAH9
        api_key = ''

        # long text generation
        model = "moonshot-v1-32k"
        # model = "moonshot-v1-8k"

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
        )
    elif args.model == 'qwen3':
        # xzyao-BLAH9
        api_key = ''

        # long text generation
        model = "qwen-plus-2025-04-28"

        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    else:
        raise ValueError(args.model + ' wrong.')

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")

    print(f'---------model: {model}---------')
    print(f'temperature: {args.temperature}, top-p: {args.top_p}.')
    print('---------------------------------')

    # Step-2 Run: repeat time
    prompt = read_doc(prompt_file)
    sent = read_doc(sent_file)

    print(f'processing {os.path.basename(sent_file)} & {os.path.basename(prompt_file)}.')

    start_time = time.time()
    for repeat in range(1, repeat_time + 1):

        print(f'repeating-{repeat}.')
        save_file = os.path.join(dir_path, f'{repeat}.txt')

        if os.path.exists(save_file) and os.path.getsize(save_file) > 0:
            print(f'"{save_file}" exists and is not empty.')
            continue

        with open(save_file, 'w') as wf:
            stream = client.chat.completions.create(
                model=model,
                max_tokens=8128,
                messages=[ {"role": "system",
                            "content": prompt},
                           {
                               "role": "user",
                               "content": sent,
                           }

                           ],
                # extra_body={"enable_thinking": False},
                # from 0 to 1
                temperature=args.temperature,
                top_p=args.top_p,
                # max_tokens=1,
                # only one response
                # n=1
                # todo: json format
                response_format={
                    'type': 'json_object'
                }
            )

            result = stream.choices[ 0 ].message.content

            wf.write(f'{str(result)}\n\n')

        print(f'{save_file} saved.')

        print(f'sleeping 30s.')
        time.sleep(30)
    end_time = time.time()
    print(f'time cost: {end_time - start_time:.4f}s.')


if __name__ == '__main__':
    main()
