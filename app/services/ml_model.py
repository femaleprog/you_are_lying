# app/services/ml_model.py

import os
from dotenv import load_dotenv
from mistralai import Mistral
from typing import Tuple, Dict

load_dotenv()
class MistralService:
    def __init__(self, agent_id: str = "ag:9518eebd:20250201:untitled-agent:74490c24"):
        
        api_key = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=api_key)
        self.agent_id = agent_id

    def get_llm_feedback(self, story: str) -> Tuple[bool, str]:
        """
        Uses the Mistral agent to analyze the story.
        The prompt includes a structured explanation of the four criteria:
          1. Personal Context
          2. Causal Coherence
          3. Sensory Details
          4. Specificity
        Few-shot examples are provided to guide the analysis.
        """
        # Build the structured prompt with few-shot examples.
        prompt = f"""
You are a narrative analysis expert. Evaluate the following story based on four criteria:

1. **Personal Context:**  
   Identify and list specific personal details provided by the speaker (e.g., background, experiences, personal anecdotes).

2. **Causal Coherence:**  
   Explain whether the story logically connects events with clear cause-and-effect relationships. Note any missing links or inconsistencies, and pay attention to contradictions

3. **Sensory Details:**  
   Identify any descriptions that appeal to the senses (sight, sound, smell, touch, taste) and comment on their richness and effectiveness.

4. **Specificity:**  
   Highlight specific details in the narrative (such as names, places, times, and other concrete information) and evaluate whether the description is detailed or vague.

Below are examples of how to analyze a story:

Example 1:

Ever since I was a child, I’ve prided myself on my punctuality and carefully structured routines. I’m known among my friends and colleagues as someone who plans every detail of my day, which has always helped me maintain a clear head and a sense of order.
On the evening of the incident, my routine was followed to the letter. I left the library where I work as the city’s research librarian at exactly 5:30 PM after finishing an engaging community reading session. At 5:35 PM, I boarded bus number 24 from the stop outside the library. The bus ride was smooth and predictable, and I arrived at my next destination—the downtown art museum—at precisely 5:50 PM.
I had a scheduled, private tour of the museum’s new Renaissance sculpture exhibit that evening. As soon as I entered, I was struck by the cool marble floors reflecting the soft, golden lighting overhead, and the hushed murmurs of other early visitors. The curator, Mr. Edwards, greeted me warmly and noted my arrival in a logbook at 5:52 PM. I spent exactly one hour immersed in the exhibit; every detail was crisp in my memory—the intricate chiseling of the marble, the subtle scent of polished stone, and the quiet reverence of the gallery space. Mr. Edwards confirmed my departure with his signature at exactly 6:52 PM.
After the tour, I took a leisurely ten-minute walk along the cobblestone streets, enjoying the crisp autumn air and the gentle rustle of fallen leaves underfoot. At 7:05 PM, I reached La Bella Vita, my favorite Italian restaurant, renowned for its authentic cuisine. I sat at my usual corner table, where the ambient light softly illuminated the rustic decor. The restaurant was alive with the sound of clinking glasses and low, warm conversations. I ordered a plate of fresh pasta with basil pesto, and the rich aroma of garlic and olive oil immediately filled my senses. My meal was as satisfying as always, complemented by the faint strains of live acoustic guitar playing in the background.
I settled my bill with a credit card, and the transaction was recorded at exactly 7:45 PM. Not wanting to miss any part of my meticulously planned evening, I boarded the evening bus at 7:50 PM from the stop right outside the restaurant and arrived back at my apartment by 8:10 PM. Every detail—from the bus timetables and the signed log at the museum to the credit card receipt at the restaurant—can be precisely verified. My routine was maintained flawlessly, and all records align perfectly with the sequence of events as I experienced them.
Analysis:
Personal context is present : The person values punctuality and structured routines, planning every detail of their day.
Works as the city's research librarian and enjoys engaging in community activities, like reading sessions.
Appreciates art, as evidenced by their interest in a private tour of a Renaissance sculpture exhibit.
Enjoys autumn walks and has a favorite Italian restaurant, La Bella Vita, where they appreciate the ambiance and cuisine.
Has a keen memory for details and enjoys the sensory experiences of their surroundings (like the scent of polished stone, the aroma of garlic and olive oil, and live acoustic guitar music).
Is meticulous and organized, as seen in how they keep track of timings and receipts. Sensory details : at least 3 sensory details are present : Sight
Golden lighting in the museum reflecting off cool marble floors.
The intricate chiseling of Renaissance sculptures.
The logbook where the curator noted the arrival time.
Rustic décor in La Bella Vita, softly illuminated by ambient light.
The cobblestone streets with fallen autumn leaves.
The bus timetables, receipts, and logs meticulously aligning with the routine.
Sound
Hushed murmurs of visitors in the museum.
Gentle rustle of fallen leaves underfoot.
The clinking of glasses and warm conversations in the restaurant.
The faint strains of live acoustic guitar playing in the background.
Smell
The subtle scent of polished stone in the museum.
The rich aroma of garlic and olive oil from the fresh pasta.
Touch
The cool, smooth marble floors underfoot.
The crisp autumn air during the evening walk.
Taste
Basil pesto pasta, rich with the flavors of garlic and olive oil. and it is specific in time and location : Time Specificity
5:30 PM – Leaves the library after a community reading session.
5:35 PM – Boards bus number 24 outside the library.
5:50 PM – Arrives at the downtown art museum.
5:52 PM – Check-in time recorded by the museum curator, Mr. Edwards.
6:52 PM – Leaves the museum, confirmed by the curator’s signature.
7:05 PM – Arrives at La Bella Vita, a favorite Italian restaurant.
7:45 PM – Pays for the meal, with a recorded credit card transaction.
7:50 PM – Boards the evening bus from the restaurant.
8:10 PM – Arrives home.
Location Specificity
Library – Works as a research librarian and leaves at a fixed time.
Bus stop – Takes a specific bus (No. 24) from the same location each day.
Downtown art museum – Attends a private tour of a Renaissance sculpture exhibit.
Cobbled streets – Walks through a specific autumn setting with leaves underfoot.
La Bella Vita – Favorite Italian restaurant, known for authentic cuisine, rustic décor, and live acoustic music.
Home – Returns by bus, arriving at precisely 8:10 PM.
Logical Flow & Causal Links
Leaving the Library (5:30 PM) → Taking the Bus (5:35 PM)

It makes sense that the character would take bus 24 from outside the library to reach the museum on time.
Bus Ride (5:35 PM → 5:50 PM) → Arriving at the Museum

The bus ride is smooth and predictable, aligning with the scheduled arrival time at 5:50 PM.
Museum Visit (5:52 PM → 6:52 PM)

The timeline holds because the curator logs their arrival and departure.
No contradiction in their engagement with the Renaissance sculpture exhibit.
Walking to Restaurant (6:52 PM → 7:05 PM)

A 10-minute walk is reasonable given the described cobblestone streets and autumn air.
The extra 3 minutes (total of 13 minutes) might be due to a slow walk, pausing to enjoy the atmosphere.
Dinner at La Bella Vita (7:05 PM → 7:45 PM)

The meal, conversation, and ambiance details logically fit within a 40-minute dining experience.
The credit card receipt at 7:45 PM confirms the timeline.
Taking the Bus (7:50 PM) → Arriving Home (8:10 PM)

A 20-minute commute is realistic.
The bus stop is right outside the restaurant, ensuring minimal transition time.
Checking for Contradictions
No overlapping or conflicting timestamps – Every event follows a consistent and realistic timeline.
No physically impossible transitions – All travel times align with reasonable walking and bus commute times.
No unexplained gaps – Every minute is accounted for, from bus schedules to meal timing.
Potential Areas to Verify Further
While no clear contradictions exist, the following could be double-checked for absolute consistency:

Bus Timetables – Are there slight variations in real-world bus arrival times? The precision suggests a rigid schedule, but minor delays could affect the sequence.
Restaurant Service Speed – A 40-minute dinner with ordering, eating, and paying is reasonable, but would depend on how quickly the food is served.
Example 2:
Story: "I was at the party last night. It was fun, and I enjoyed myself."
Analysis:
- Personal Context: The narrative offers minimal personal context, simply stating attendance.
- Causal Coherence: There is no clear sequence or explanation of events.
- Sensory Details: The story lacks sensory descriptions.
- Specificity: The narrative is vague with no specific details provided.

Now, analyze the following story:
Story: "{story}"
        """

        # Call the Mistral agent with the prompt.
        response = self.client.agents.complete(
            agent_id=self.agent_id,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        # Get the content from the first message choice.
        feedback = response.choices[0].message.content

        # A simple heuristic to determine if the narrative is coherent.
        is_coherent = "coherent" in feedback.lower() and "not coherent" not in feedback.lower()
        return is_coherent, feedback
