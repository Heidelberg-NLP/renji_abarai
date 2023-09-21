from pathlib import Path
from argparse import Namespace
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import json
import openai
from time import sleep

openai.api_key_path = Path("path/to/openai_api_key")

"""
This script generates predictions from ChatGPT for quality and stance.
"""

def load_data(fn:Path):
    """
    :param fn: filename
    :return: a dataframe with the following columns:
    for dev:
        `q_id`
        `doc_id`
        `rel` is the relevance score
        `qual` is the quality score
        `query` is the query
        `concl`
        `prem` is the document
        `stance`
    for test:
        `qid` (gets renamed to `q_id`)
        `query`
        `docno` (gets renamed to `doc_id`)
        `score`
        `title_text`
        `html_plain`
        `truncated_html_plain` (gets renamed to `prem`)
        `rank`
    """
    df = pd.read_csv(fn, sep="\t")

    if 'qid' in df.columns:
        df = df.rename(columns={'qid': 'q_id', 'docno': 'doc_id', 'truncated_html_plain': 'prem'})

    return df

def parse_prompt(text:str):
    """
    converts a text to a list of messages
    """
    if not (('<USER>' in text) or ('<CHAT>' in text)):
        print('no chat, just one prompt')
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": text},
        ]
        return messages

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    user = False
    chat = False
    for line in text.split('\n'):
        if '<USER>' == line.strip():
            user = 1 - user
            if user: # user part started
                assert not chat
                tmp_line = ''
            else:
                messages.append({"role": "user", "content": tmp_line.strip()})
            continue
        if '<CHAT>' == line.strip():
            chat = 1 - chat
            if chat:  # chat part started
                assert not user
                tmp_line = ''
            else:
                messages.append({"role": "assistant", "content": tmp_line.strip()})
            continue
        tmp_line += line + '\n'
    return messages


def get_prediction(
        text:str, 
    ):
    """
    Get the prediction from the model
    :param text: input text
    """
    messages = parse_prompt(text)

    reply = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",  # 4097 tokens max
        messages=messages,
    )

    return reply

def load_prompt(
        fn:Path,
    ):
    """
    Load the prompt. The prompt should contain `<TEXT>` which will be replaced by the input text. Optionally in can contain `<QUERY>` which will be replaced by the query.
    `<USER>` and `<CHAT>` are used to indicate the start of the user and chat part respectively.
    :param fn: filename
    :return: a string
    """
    prompt = fn.read_text()
    return prompt

def main(args:Namespace):
    args.fn_out.parent.mkdir(exist_ok=True, parents=True)
    args.out_dir.mkdir(exist_ok=True, parents=True)

    print('load data')
    data = load_data(fn=args.fn_data)

    print('load prompt')
    prompt = load_prompt(fn=args.fn_prompt)

    print('iterate over data')
    if args.fn_out_label:
        counter = defaultdict(lambda:0)
        labels = []

    for i, (_, d) in tqdm(enumerate(data.iterrows()), total=len(data)):
        document = d.prem
        query = d.query
        tmp_prompt = prompt.replace('<TEXT>', document).replace('<QUERY>', query)

        sleep(0.5)  # to avoid the rate limit

        if args.fn_out_label:
            if args.task == 'quality':
                label = str(d.qual)
            elif args.task == 'stance':
                label = str(d.stance)

            if counter[label] >= 5:
                continue
            counter[label] += 1
            labels.append(label)

        try:
            prediction = get_prediction(tmp_prompt)
            pred_text = prediction.choices[0].message.content
        except:  # if the number of tokens is too long, then the OpenAI API will return an error
            try:  # truncate document
                print('error 1')
                tmp_prompt = prompt.replace('<TEXT>', document[:1500*4]).replace('<QUERY>', query)
                prediction = get_prediction(tmp_prompt)
                pred_text = prediction.choices[0].message.content
            except:  # truncate document even more
                print('error 2')
                tmp_prompt = prompt.replace('<TEXT>', document[:500*4]).replace('<QUERY>', query)
                prediction = get_prediction(tmp_prompt)
                pred_text = prediction.choices[0].message.content

        # save the prediction
        json.dump(prediction, Path(args.out_dir, f'index={i}_queryid={d.q_id}_docid={d.doc_id}_prediction.json').open('w'), indent=2)
        Path(args.out_dir, f'index={i}_queryid={d.q_id}_docid={d.doc_id}_prompt.json').open('w').write(tmp_prompt)

        with args.fn_out.open('a') as f:
            f.write(pred_text)
            f.write('\n\n###\n\n')

    if args.fn_out_label:
        json.dump(labels, args.fn_out_label.open('w'), indent=2)


if __name__ == "__main__":
    use_custom_stopwords = True  # False to use official puss-in-boots baseline. True to use Renji-Abarai baseline

    task = 'quality'
    # task = 'stance'

    # prompt_name = 'instruction_examples'
    # prompt_name = 'examples'
    # prompt_name = 'examples-chat'
    # prompt_name = 'instruction_examples-chat'
    prompt_name = 'instruction_examples-chat_short'
    # prompt_name = 'instruction'
    # prompt_name = 'neither'
    # prompt_name = 'examples-unbalanced'
    # prompt_name = 'instruction_examples-unbalanced'
    # prompt_name = 'instruction-unbalanced'
    # prompt_name = 'instruction-unbalanced_examples-unbalanced'
    # prompt_name = 'instruction-unbalanced_examples'
    # prompt_name = 'instruction-neutral_examples-short'
    # prompt_name = 'instruction-yesno_examples'
    # prompt_name = 'examples-excellent'
    # prompt_name = 'neither-excellent'
    # prompt_name = 'instruction-excellent'
    # prompt_name = 'instruction-excellent_examples-excellent'

    # parameter for the dev set
    # args = Namespace(
    #     fn_data=Path('data/qrels_args_docs_val_subsample-60.tsv'), 
    #     fn_prompt=Path(f'data/prompts/{task}/{prompt_name}.txt'),
    #     fn_out=Path(f'data/results/{task}/ChatGPT/{prompt_name}.txt'),
    #     fn_out_label=Path(f'data/results/{task}/ChatGPT/labels_{prompt_name}.json'),
    #     out_dir=Path(f'data/results/{task}/ChatGPT/full_outputs/{prompt_name}'),
    #     task=task,
    # )

    # parameter for the test set
    args = Namespace(
        fn_data=Path(f'results-test/corpus/chatnoir_10{"_custom_stopw_lemmas" if use_custom_stopwords else ""}.tsv'),  # filename of the corpus
        fn_prompt=Path(f'data/prompts/{task}/{prompt_name}.txt'),  # fn of the prompt
        fn_out=Path(f'results-test/ChatGPT/{task}{"" if use_custom_stopwords else "_from-baseline"}/{prompt_name}__run1.txt'),  # fn of the output
        fn_out_label=None,  # fn where the gold-labels are saved. This is only used for the dev set. To save costs, when this is set (i.e. not `None`), then automatically only a few instances per label (according to the gold labels) are processed.
        out_dir=Path(f'results-test/ChatGPT/{task}{"" if use_custom_stopwords else "_from-baseline"}/full_outputs/{prompt_name}__run1'),  # directory where the full outputs are saved
        task=task,  # task that is solved (either `quality` or `stance`)
    )
    print(args)
    print('wait 10 seconds to give time to quit in case args are wrong...')
    sleep(10)
    print('start')

    main(args)
