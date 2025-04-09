import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

data = {
    "text": [
        # Healthy samples (0-99)
        "I love painting and gardening. It helps me feel calm and happy every day.",
        "My daughter visited me yesterday. We had coffee and talked about her job.",
        "I finished reading a book last night. The story was about a detective solving a mystery.",
        "Every morning, I take a walk in the park. The fresh air helps me start the day right.",
        "I cooked pasta for dinner tonight. I used a recipe from my favorite cookbook.",
        "The weather is beautiful today. I think I'll sit outside and enjoy the sunshine.",
        "I called my friend earlier. We planned to meet for lunch next weekend.",
        "I cleaned the house this morning. Now everything looks tidy and organized.",
        "I watched a documentary about whales. They're such fascinating creatures.",
        "I wrote a letter to my grandson. I told him about my trip to the mountains.",
        "I listen to classical music while working. It helps me concentrate better.",
        "I planted new flowers in my garden. They'll bloom in a few weeks.",
        "I went to the gym today. I've been trying to exercise three times a week.",
        "I baked cookies this afternoon. The kitchen smells wonderful now.",
        "I volunteered at the library. I helped organize the children's book section.",
        "I attended a concert last weekend. The orchestra played Beethoven's symphonies.",
        "I finished knitting a scarf. I'll give it to my neighbor for her birthday.",
        "I solved a crossword puzzle today. It took me about twenty minutes.",
        "I met my brother for breakfast. We talked about our childhood memories.",
        "This morning I made pancakes for the family. Everyone enjoyed them with maple syrup.",
        "I rearranged the living room furniture yesterday. The new layout feels more spacious.",
        "My book club discussed a fascinating novel last week. We all had different interpretations.",
        "I learned how to make sourdough bread during quarantine. Now it's my specialty.",
        "The grandkids came over on Saturday. We built a fort with blankets in the living room.",
        "I've been practicing Spanish for 30 minutes each day. My vocabulary is improving steadily.",
        "Yesterday I visited the art museum downtown. They had an excellent Impressionist exhibit.",
        "My husband and I celebrated our anniversary at our favorite Italian restaurant.",
        "I started a journal to document my travels. It's fun to revisit old memories.",
        "The neighborhood had a block party last night. We met several new families.",
        "I completed a 1000-piece jigsaw puzzle of the Eiffel Tower. It took me three days.",
        "My sister and I video chat every Sunday. We share recipes and talk about our gardens.",
        "I signed up for a pottery class at the community center. The first session is next week.",
        "The roses in my backyard are blooming beautifully this season. The red ones are my favorite.",
        "I organized all my old photos into albums. It was wonderful reminiscing about the past.",
        "My son taught me how to use video editing software. I'm making a family vacation movie.",
        "I've been waking up early to watch the sunrise. The colors over the lake are breathtaking.",
        "The book I'm reading has such an intriguing plot. I can't wait to see how it ends.",
        "I made homemade lemonade yesterday. The perfect drink for a warm summer afternoon.",
        "My friend and I went birdwatching in the nature preserve. We spotted seven different species.",
        "I've been keeping track of my daily steps. Yesterday I walked over 10,000 steps!",
        "The farmers market had fresh strawberries today. I bought two baskets to make jam.",
        "I taught my granddaughter how to play chess. She's learning quickly and beat me twice already.",
        "The yoga class at the senior center has improved my flexibility significantly.",
        "I repaired the leaky faucet in the kitchen. It feels good to fix things myself.",
        "My daughter sent me flowers for Mother's Day. They're bright yellow daffodils.",
        "I've been experimenting with watercolor painting. Landscapes are my favorite subject.",
        "The historical society lecture was fascinating. I learned so much about our town's origins.",
        "I baked an apple pie using my grandmother's recipe. The cinnamon aroma filled the whole house.",
        "My walking group meets every Tuesday morning. We explore different trails around the city.",
        "I finished knitting mittens for all my grandchildren. Just in time for winter too.",
        "The documentary about coral reefs was eye-opening. I never realized how fragile they are.",
        "I organized a surprise birthday party for my husband. All our children came home for it.",
        "My tomato plants are producing so much fruit. I've been giving baskets to all my neighbors.",
        "I started learning to play the ukulele. My fingers are sore but it's so much fun.",
        "The library book sale was this weekend. I found several classic novels for my collection.",
        "I wrote down all my favorite family recipes in a notebook for my daughter.",
        "The meteor shower last night was spectacular. We stayed up late to watch it from the backyard.",
        "I've been meditating for 15 minutes each morning. It helps me stay centered throughout the day.",
        "My old college friends and I reunited for lunch. We laughed about our youthful adventures.",
        "I created a scrapbook of my travels through Europe. The memories came flooding back.",
        "The community theater production was excellent. The lead actress had a magnificent voice.",
        "I've been collecting seashells on my beach walks. I'm going to make a wind chime with them.",
        "My grandson helped me set up a bird feeder. We've already seen cardinals and blue jays visit.",
        "I perfected my chili recipe after many attempts. The secret is a dash of cocoa powder.",
        "The astronomy club meeting was fascinating. We learned about the new Mars rover mission.",
        "I donated several boxes of books to the school fundraiser. It felt good to declutter.",
        "My hydrangeas won first prize at the garden show. All that pruning paid off!",
        "I've been writing short stories about my childhood. My family finds them entertaining.",
        "The cooking class taught us how to make authentic Thai curry. The flavors were incredible.",
        "I framed all the children's artwork from over the years. The hallway is now a gallery.",
        "The nature photography exhibit inspired me to take better pictures of my garden.",
        "I organized a family genealogy project. We discovered ancestors dating back to the 1700s.",
        "My morning routine includes stretching exercises and a healthy smoothie.",
        "The antique shop had a beautiful set of teacups. I bought them to use for special occasions.",
        "I volunteered at the animal shelter yesterday. Playing with the puppies was so joyful.",
        "My book about local wildflowers is nearly complete. I've been working on it for two years.",
        "The grandchildren helped me plant a vegetable garden. They're excited to watch it grow.",
        "I've been practicing calligraphy. My holiday cards will look extra special this year.",
        "The community choir performance went wonderfully. My solo received a standing ovation.",
        "I restored an old rocking chair I found at a flea market. It's now my favorite reading spot.",
        "The homemade bread turned out perfectly crusty. I used a new yeast strain from France.",
        "I cataloged all my record collection alphabetically. It took all weekend but was worth it.",
        "The knitting circle at the library is so welcoming. We share patterns and life stories.",
        "I planned a picnic at the botanical gardens. The roses were in full bloom and fragrant.",
        "My crossword puzzle streak continues - 45 days in a row completing the Sunday challenge.",
        "The grandchildren and I built a birdhouse together. We painted it bright blue and red.",
        "I've been researching my family's immigration history. The stories are fascinating.",
        "The quilting project I started last winter is finally finished. It's a gift for my niece.",
        "I organized all my spices alphabetically and bought matching jars. The kitchen looks neater.",
        "The memoir writing class has been therapeutic. Sharing stories with others is healing.",
        "I perfected my grandmother's biscuit recipe after many attempts. They're now just as fluffy.",
        "The local history museum recruited me as a volunteer guide. I love sharing our town's story.",
        "I've been collecting vintage postcards from places I've visited. They make a beautiful collage.",
        "The piano in the community center needed tuning. I volunteered to play it afterward.",
        "My herb garden is thriving this season. The basil is especially fragrant and plentiful.",
        "I compiled all my mother's handwritten recipes into a book for the whole family.",
        "The walking tour of historic homes was enlightening. The architecture was breathtaking.",
        "I've been journaling about my gardening experiences. Maybe I'll publish it someday.",
        "The grandchildren helped me bake Christmas cookies. We made stars, trees, and snowmen.",
        "I organized a neighborhood clean-up day. Everyone pitched in and we filled ten trash bags.",
        "Uh, so like... I don't remember what I was doing. Maybe I went to the place, um, yesterday?",
        "He, um, he was there... and then, uh, she said something about, I don't know, the dog maybe.",
        "I went to the doctor today. He said my memory has gotten worse over the past year.",
        "Umm... I don't remember names. It's hard to find the right words sometimes.",
        "I was supposed to... uh... do something, but I can't recall what it was.",
        "The thing is... um... where did I put my keys? I just had them a minute ago.",
        "She told me her name, but... uh... it's gone now. I think it started with a 'J'?",
        "I went to the... uh... store? No, maybe it was the park. I'm not sure.",
        "I keep forgetting appointments. My daughter had to remind me about the... um... the thing.",
        "What was I saying? Oh, right... um... never mind, I lost my train of thought.",
        "I tried to cook, but I... uh... forgot the recipe halfway through.",
        "I saw my friend, but... um... I couldn't remember how I knew her.",
        "Where is my wallet? I thought I left it on the... uh... table? Or was it the kitchen?",
        "I wanted to tell you something important, but... uh... it slipped my mind.",
        "I got lost driving home. The streets looked... um... unfamiliar.",
        "I forgot to take my pills again. I think it's the third time this week.",
        "I couldn't finish my sentence because... uh... the word just disappeared.",
        "I don't remember eating lunch, but my plate is... um... empty.",
        "I stared at the phone, but... uh... I couldn't recall who I was calling.",
        "I left the stove on yesterday. My son said it's becoming a... um... problem.",
        "The... the person who lives next door... what's her name again? She was just here.",
        "I was watching... um... that show with the doctor... what's it called? It's popular.",
        "My... my... you know, the one who married my son... can't think of the word... daughter-in-law!",
        "I went to get something from the... the cold box in the kitchen... refrigerator!",
        "I know I had an appointment today, but... uh... can't remember where or when.",
        "What day is it today? I thought it was Tuesday but my calendar says Wednesday.",
        "I put my glasses down somewhere and now... um... they're just gone.",
        "The story I was telling... um... what was the point I was trying to make?",
        "I called you earlier because... uh... oh dear, I've completely forgotten why.",
        "I was in the middle of a task when... uh... what was I doing again?",
        "The grandkids came over and we... um... did something fun but I can't recall what.",
        "I know this person well but... uh... their name is on the tip of my tongue.",
        "I was reading a book but... um... can't remember the title or what it's about.",
        "I went to the place with all the books... library! That's what it's called.",
        "My daughter called to tell me something important but... uh... it's slipped my mind.",
        "I was going to make tea but... um... now I can't find the kettle I just had.",
        "The program I watch every night... um... it's not coming to me right now.",
        "I know I needed to buy something at the store but... uh... can't remember what.",
        "I started cleaning but... um... got distracted and forgot what I was doing.",
        "The neighbor asked me about... uh... some event, but I don't recall the details.",
        "I was telling a joke but... um... forgot the punchline halfway through.",
        "My medication schedule is... uh... I think I took them but maybe I didn't.",
        "The restaurant we went to last week... um... can't remember what I ordered.",
        "I put something important in a safe place but... uh... now I can't find it.",
        "The name of that thing you use to open cans... um... it starts with a 'c'...",
        "I was going to water the plants but... uh... did I already do that today?",
        "The movie we watched recently... um... the plot is completely gone from my mind.",
        "I know I had a question for you but... uh... it's disappeared from my thoughts.",
        "The street where I lived as a child... um... the name is just not coming to me.",
        "I was going to call someone important but... uh... can't remember who.",
        "The password I've used for years... um... suddenly I can't recall it.",
        "I made a list of things to do but... uh... now I can't find the list.",
        "The relative who visited last month... um... can't remember who it was.",
        "I was in the middle of a conversation when... uh... what were we talking about?",
        "The item I use to chop vegetables... um... knife! That's the word I wanted.",
        "I know I parked the car somewhere but... uh... can't remember where.",
        "The holiday we celebrated last... um... was it Christmas or Thanksgiving?",
        "I was going to take a shower but... uh... did I already take one today?",
        "The author of that book I liked... um... his name is right there but I can't...",
        "I was supposed to meet someone but... uh... can't recall who or where.",
        "The thing you tell time with... um... watch! That's what I was trying to say.",
        "I know I had a doctor's appointment but... uh... can't remember when.",
        "The food I ate for breakfast... um... did I even have breakfast today?",
        "I was going to pay a bill but... uh... which one was it again?",
        "The place where you buy groceries... um... supermarket! The word escaped me.",
        "I know I locked the door but... uh... now I'm not sure if I actually did.",
        "The show I watch every morning... um... can't recall the name right now.",
        "I was going to feed the pet but... uh... did I already do that?",
        "The relative who called yesterday... um... was it my sister or my niece?",
        "I know I wrote down an important date but... uh... where did I put that note?",
        "The appliance that keeps food cold... um... refrigerator! It took me a moment.",
        "I was supposed to turn something off but... uh... was it the oven or the lights?",
        "The holiday coming up soon... um... is it someone's birthday? I can't recall.",
        "I know I put my purse somewhere safe but... uh... now I can't find it.",
        "The program where people answer questions to win prizes... um... game show!",
        "I was going to take my medication but... uh... did I already take it?",
        "The street where my best friend lives... um... the name is on the tip of my tongue.",
        "I know I had a good idea earlier but... uh... now it's completely gone.",
        "The thing you sit on outside... um... lawn chair! The word wouldn't come.",
        "I was supposed to buy something at the store but... uh... what was it again?",
        "The relative who had a baby recently... um... can't remember who it was.",
        "I know I turned off the stove but... uh... maybe I should check again.",
        "The show with the detectives solving crimes... um... what's it called?",
        "I was going to write a letter but... uh... who was it for again?",
        "The thing you use to dry off after a shower... um... towel! It took me a while.",
        "I know I had an important paper but... uh... where did I put it?",
        "The holiday when we give gifts... um... is it Christmas or Valentine's Day?",
        "I was supposed to call someone back but... uh... can't remember who.",
        "The place where you borrow books... um... library! Why couldn't I think of that?",
        "I know I scheduled an appointment but... uh... can't recall with whom.",
        "The food I was going to cook for dinner... um... what was the recipe again?",
        "I was going to water the plants but... uh... did I already do that?",
        "The relative who's coming to visit... um... is it my son or my brother?",
        "I know I had a question about something but... uh... it's gone now.",
        "The thing you use to cut paper... um... scissors! The word escaped me.",
        "I was supposed to meet someone for lunch but... uh... who was it again?",
        "The program about cooking competitions... um... what's that show called?",
        "I know I put my glasses somewhere but... uh... now I can't find them.",
        "The street where my doctor's office is... um... can't remember the name.",
        "I was going to take a walk but... uh... did I already go today?"
    ],
    "label": [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Healthy (0-19)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Healthy (20-39)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Healthy (40-59)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Healthy (60-79)
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # Healthy (80-99)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Cognitive (100-119)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Cognitive (120-139)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Cognitive (140-159)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # Cognitive (160-179)
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1   # Cognitive (180-199)
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Verify lengths
print(f"Text length: {len(df['text'])}")
print(f"Label length: {len(df['label'])}")

# Preprocessing function
def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Apply preprocessing
df["clean_text"] = df["text"].apply(preprocess_text)

# Feature extraction function
def extract_features(text):
    doc = nlp(text)
    num_tokens = len(doc)
    num_sents = len(list(doc.sents))
    avg_sent_len = num_tokens / num_sents if num_sents > 0 else 0

    pos_counts = Counter([token.pos_ for token in doc])
    pronoun_count = sum(1 for token in doc if token.pos_ == "PRON")
    noun_count = pos_counts.get("NOUN", 0)
    verb_count = pos_counts.get("VERB", 0)

    return pd.Series({
        "avg_sent_len": avg_sent_len,
        "num_pronouns": pronoun_count,
        "num_nouns": noun_count,
        "num_verbs": verb_count
    })

# Extract features
feature_df = df["text"].apply(extract_features)
df = pd.concat([df, feature_df], axis=1)

# Visualization
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
sns.boxplot(x="label", y="avg_sent_len", data=df)
plt.title("Average Sentence Length vs Label")

plt.subplot(1, 3, 2)
sns.boxplot(x="label", y="num_pronouns", data=df)
plt.title("Number of Pronouns vs Label")

plt.subplot(1, 3, 3)
sns.boxplot(x="label", y="num_nouns", data=df)
plt.title("Number of Nouns vs Label")

plt.tight_layout()
plt.show()

# Save to CSV
df.to_csv("/Users/rishabhbhargav/PycharmProjects/Cognitive_NLP/data/processed/simulated_features.csv", index=False)