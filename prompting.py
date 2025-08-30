from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

zero_shot_prompt = """
Tell me about indian prime minister manmohan singh
"""

one_shot_prompt = """
Tell me about indian prime minister manmohan singh
Example 1:
Donald John Trump (born June 14, 1946) is an American businessman, media personality, and politician who is the 47th president of the United States since 2025. Before, was the 45th president from 2017 to 2021. He is a member of the Republican Party. Before becoming president, he was a businessman and television personality.
"""

few_shot_prompt = """
Tell me about indian prime minister manmohan singh
Example 1:
Donald John Trump (born June 14, 1946) is an American businessman, media personality, and politician who is the 47th president of the United States since 2025. Before, was the 45th president from 2017 to 2021. He is a member of the Republican Party. Before becoming president, he was a businessman and television personality.
Example 2:
Abraham Lincoln (February 12, 1809 - April 15, 1865) was the 16th president of the United States, serving from 1861 until his assassination in 1865. He led the United States through the American Civil War, defeating the Confederate States of America and playing a major role in the abolition of slavery.
"""

few_shot_chain_of_thought_prompting = """
Q: At Gada electronics, Jethalal decides marks up his electronic goods 50% on cost price to make good profit. To further entice customers, he offers subsequent discounts of 10% + 10% on the marked price. What is the profit percentage Jethalal ultimately earns from his sales?

A: Lets think step by step
step 1: Let the cost price (CP) be ₹100.
step 2: Marked price (MP) = 100 + 50% of 100 = ₹150.
step 3: Selling price (SP) = MP * 0.9 * 0.9 = 121.5
step 4: Profit % = [(SP - CP) / CP] * 100 = [(121.5 - 100) / 100] * 100 = 21.5%.

Q: You are a contractor tasked with creating two unique exercise areas in a recreational park. The first area is a square-shaped field, which has been finalized with a side length of 55 meters. The second area is a circular running track that must have the same perimeter as the square field. Determine the radius of this circular track.
"""

zero_shot_chain_of_thought_prompting = """
Q: You are a contractor tasked with creating two unique exercise areas in a recreational park. The first area is a square-shaped field, which has been finalized with a side length of 55 meters. The second area is a circular running track that must have the same perimeter as the square field. Determine the radius of this circular track.
A: lets think step by step..
"""

test_prompt = """
When I was 6 my sister was half my age. Now
I'm 70 how old is my sister?
"""

# response = llm.invoke(zero_shot_prompt)
# response = llm.invoke(one_shot_prompt)
# response = llm.invoke(few_shot_prompt)
# response = llm.invoke(few_shot_chain_of_thought_prompting)
# response = llm.invoke(zero_shot_chain_of_thought_prompting)
response = llm.invoke(test_prompt)
print(response.content)

# https://www.youtube.com/watch?v=wNKTBCuuMUU
# https://www.promptingguide.ai/techniques/cot



