import openai as ai
from config import Settings

settings = Settings()
def GPTFunction(document_content,prompt):
    # Replace YOUR_API_KEY with your OpenAI API key
    ai.api_key = settings.GPT_API_KEY

    model_engine = "text-davinci-003"
    #print(prompt)
    prompt = prompt + str( {document_content} )

    # Set the maximum number of tokens to generate in the response
    max_tokens = 1024

    # Generate a response
    completion = ai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.5,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )
    ChatGPT_output = str(completion.choices[0].text)
    return ChatGPT_output