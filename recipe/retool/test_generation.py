from openai import OpenAI


client = OpenAI(base_url="http://210.28.135.36:8000/v1", api_key="EMPTY")


TOOLS = [
    {
        'function': {
            'description': 'Execute Python code to perform calculations, data analysis, or other computational tasks.', 
            'name': 'code_interpreter', 
            'parameters': {
                'properties': {
                    'code': {
                        'description': 'The Python code to be executed. This can include calculations, data manipulation, or any valid Python code.', 
                        'type': 'string'
                    }, 
                    'language': {
                        'default': 'python', 
                        'description': 'The programming language of the code. Currently only Python is supported.', 
                        'type': 'string'
                    }
                }, 
                'required': ['code'], 
                'type': 'object'
            }
        }, 
        'type': 'function'
    },
]



problem = """
Alice and Bob play the following game. A stack of $n$ tokens lies before them. The players take turns with Alice going first. On each turn, the player removes either $1$ token or $4$ tokens from the stack. Whoever removes the last token wins. Find the number of positive integers $n$ less than or equal to $2024$ for which there exists a strategy for Bob that guarantees that Bob will win the game regardless of Alice's play.
"""

system_message = "You are a helpful assistant that can solve math problems with interaction Code Interpreter by Python code."
user_message = f"""Solve the following problem step by step. You now have the ability to selectively write executable Python code to enhance your reasoning process.


**user question:** 
{problem}


Remember to place the final answer in the last part using the format: 
<answer>
\\boxed{{'The final answer goes here.'}}
</answer>
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_message},
]


response = client.chat.completions.create(
    model="/data2/lixy/VerlCoder/checkpoints/retool-multiturn-sft/retool-multiturn-sft-qwen2.5-7b-sp4-lr5e-6/global_step_42",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto",
    max_tokens=16384,
    temperature=0.7,
    top_p=0.9,
    extra_body={
        "top_k": 50,
    }
)

print(response.choices[0].message.content)
print(response.choices[0].message.tool_calls)