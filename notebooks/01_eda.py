import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

# Full dataset (over 100 examples)
data = {
    "text": [
        # Tier 0: Healthy (60 samples)
        "I love painting and gardening. It helps me feel calm and happy every day.",
        "I forgot where I parked but found the car after checking a few aisles.",
        "I couldn’t recall the actor’s name, but remembered it was from that crime show.",
        "My daughter visited me yesterday. We had coffee and talked about her job.",
        "I finished reading a book last night. The story was about a detective solving a mystery.",
        "I made breakfast this morning and then went for a walk in the park.",
        "I had a video call with my granddaughter. We talked about her school project.",
        "I enjoy crossword puzzles and finish one every night before bed.",
        "I take care of my plants every morning and water them regularly.",
        "I went to my book club meeting and shared my thoughts on the latest novel.",
        "I remember birthdays of all my children and call them on their special days.",
        "I helped my neighbor organize her garage over the weekend.",
        "I baked cookies for the community event and everyone loved them.",
        "I’ve been learning how to play the piano through online tutorials.",
        "I met my friend from high school and we reminisced about old times.",
        "I planned a trip with my spouse and remembered all the details without notes.",
        "I balanced my checkbook without any errors this month.",
        "I recognized an old colleague at the store and recalled their name instantly.",
        "I followed a new recipe and cooked dinner without mistakes.",
        "I remembered to take my vitamins every day this week.",
        "I organized my closet and remembered where everything was placed.",
        "I attended a lecture and could summarize the key points afterward.",
        "I played chess with my grandson and remembered all the rules.",
        "I recalled the name of a childhood friend I hadn’t seen in years.",
        "I kept track of all my appointments this month without reminders.",
        "I remembered the lyrics to my favorite song from decades ago.",
        "I navigated to a new restaurant without using GPS.",
        "I recalled the plot of a movie I watched years ago.",
        "I recognized a familiar face in a crowded place immediately.",
        "I remembered to water my neighbor’s plants while they were away.",
        "I recalled the name of a book I read last year without hesitation.",
        "I followed a complex discussion in a group meeting.",
        "I remembered to buy all the groceries on my list.",
        "I recalled the name of my first-grade teacher.",
        "I kept track of time without constantly checking the clock.",
        "I remembered the details of a story I heard last week.",
        "I recognized a song from just the first few notes.",
        "I recalled the name of a restaurant I visited months ago.",
        "I remembered all the ingredients for a dish I cook often.",
        "I navigated through a new city using landmarks.",
        "I recalled the birthdays of all my siblings.",
        "I remembered the plot twists in a mystery novel.",
        "I recognized an actor from a minor role in an old film.",
        "I recalled the directions to a friend’s house from years ago.",
        "I remembered the name of a childhood pet.",
        "I kept track of multiple tasks without writing them down.",
        "I recalled the details of a conversation from last month.",
        "I recognized a voice on the phone before they said their name.",
        "I remembered the rules of a board game I hadn’t played in years.",
        "I recalled the name of a street I lived on as a child.",
        "I navigated a new public transport system without issues.",
        "I remembered the plot of a TV show I watched last year.",
        "I recognized a fragrance from my childhood.",
        "I recalled the name of a teacher from high school.",
        "I remembered the sequence of a workout routine.",
        "I recognized a painting and remembered the artist’s name.",
        "I recalled the name of a childhood best friend.",
        "I remembered the steps of a dance I learned years ago.",
        "I recognized a brand from its logo alone.",
        "I recalled the name of a distant relative.",

        # Tier 1: Mild Concern (60 samples)
        "I left my keys in the fridge again, but at least I remembered where I put them this time.",
        "I told my grandson the same story twice this week—he politely didn’t mention it.",
        "I missed two doctor appointments this month because I wrote down the wrong dates.",
        "I sometimes forget what day it is but figure it out quickly.",
        "I couldn’t remember my cousin’s name during a phone call yesterday.",
        "I occasionally misplace my glasses, but I always find them eventually.",
        "I asked the same question twice during dinner without realizing it.",
        "I mixed up the names of my neighbors but caught myself right away.",
        "I had trouble remembering if I locked the door, so I double-checked.",
        "I forgot the plot of a movie I watched last week.",
        "I started writing a grocery list and realized I had already written one earlier.",
        "I lost track of time during the afternoon and missed a TV show I like.",
        "I repeated the same instruction while helping my granddaughter bake cookies.",
        "I found my phone in the laundry basket, though I remembered putting it there later.",
        "I wrote my check wrong and had to start over because I got the date confused.",
        "I forgot where I put my wallet, but found it in my jacket pocket after searching.",
        "I called my son by the dog’s name, then laughed it off.",
        "I walked into a room and forgot why I went there.",
        "I forgot the name of a common household item for a moment.",
        "I had to ask my spouse what we discussed yesterday.",
        "I left the water running but remembered to turn it off after a few minutes.",
        "I forgot to take my morning pills until noon.",
        "I repeated a joke I had already told the same group.",
        "I misplaced my shopping list but recalled most items from memory.",
        "I forgot an old friend’s name at a reunion but remembered later.",
        "I struggled to recall the name of a famous landmark.",
        "I forgot to reply to an email for several days.",
        "I mixed up the dates of two family birthdays.",
        "I had to reread a paragraph in a book because I forgot its content.",
        "I forgot the name of a restaurant I visited recently.",
        "I left my umbrella at a café but remembered where I left it.",
        "I forgot the password to an account and had to reset it.",
        "I misplaced my reading glasses multiple times in one day.",
        "I forgot to buy an item on my grocery list.",
        "I repeated a question in a conversation without noticing.",
        "I forgot the name of a movie I watched last month.",
        "I had to check my calendar to confirm an appointment.",
        "I forgot the name of a colleague I don’t see often.",
        "I left a pot on the stove longer than intended.",
        "I forgot where I stored a rarely used kitchen tool.",
        "I had to think hard to recall a childhood memory.",
        "I forgot to bring a necessary item to an event.",
        "I mixed up two similar-sounding names.",
        "I forgot the details of a recent phone call.",
        "I misplaced my keys more than once in a week.",
        "I forgot the name of a book I recently read.",
        "I had to ask for directions to a familiar place.",
        "I forgot to charge my phone overnight.",
        "I repeated a story to the same person.",
        "I forgot the name of a common spice while cooking.",
        "I misplaced my hat but found it later.",
        "I forgot the name of a famous actor.",
        "I had to look up a recipe I’ve made before.",
        "I forgot to water the plants for a day.",
        "I repeated myself in a group discussion.",
        "I forgot the name of a street near my home.",
        "I misplaced my pen but found it in my pocket.",
        "I forgot the name of a song I like.",
        "I had to think to recall a recent event.",
        "I forgot to set an alarm and overslept.",

        # Tier 2: Moderate (60 samples)
        "I put the milk in the cupboard and the teabag in my pocket.",
        "I called my daughter by my sister’s name all day and didn’t notice.",
        "I forgot how to use the TV remote I’ve had for years.",
        "I went to the... uh... store? No, maybe it was the park. I’m not sure.",
        "I left the stove on yesterday. My son said it’s becoming a... um... problem.",
        "I couldn’t remember how to get to the doctor’s office I’ve been going to for years.",
        "I forgot how to write a check yesterday and had to ask for help.",
        "I stared at the microwave for minutes trying to remember how to use it.",
        "I asked my neighbor the same question three times in one conversation.",
        "I saw a picture of my grandchild and couldn’t remember his name.",
        "I keep confusing the bathroom and kitchen light switches.",
        "I forgot how to log into my email and had to reset my password again.",
        "I went to the bank and forgot why I was there.",
        "I forgot to add water to the kettle before turning it on.",
        "I started vacuuming the floor but forgot I already finished it earlier.",
        "I put my shoes in the refrigerator and didn’t realize until later.",
        "I forgot the name of my own street when giving directions.",
        "I tried to unlock my front door with the car key.",
        "I forgot how to operate the washing machine I’ve used for a decade.",
        "I couldn’t recall my own phone number when asked.",
        "I left the groceries in the car overnight and they spoiled.",
        "I forgot the name of my best friend from college.",
        "I put salt in my coffee instead of sugar.",
        "I forgot how to use the thermostat and set it wrong.",
        "I asked my son who he was when he came to visit.",
        "I forgot the name of a common fruit at the store.",
        "I tried to pay with a library card instead of a credit card.",
        "I forgot how to turn off the oven after cooking.",
        "I put the remote control in the dishwasher.",
        "I forgot the name of my own pet.",
        "I left the front door unlocked all night.",
        "I forgot how to make a simple recipe I’ve made for years.",
        "I put the laundry detergent in the fridge.",
        "I forgot the name of my childhood home.",
        "I tried to call someone but forgot how to dial.",
        "I left the car running in the garage.",
        "I forgot the name of a close family member.",
        "I put my glasses in the freezer.",
        "I forgot how to tie my shoelaces.",
        "I left the bath running and overflowed it.",
        "I forgot the name of my favorite book.",
        "I tried to brush my teeth with shaving cream.",
        "I forgot how to use the coffee maker.",
        "I put my wallet in the pantry.",
        "I forgot the name of the current month.",
        "I left my phone in the mailbox.",
        "I forgot how to get home from a familiar place.",
        "I put the newspaper in the oven.",
        "I forgot the name of my own child for a moment.",
        "I tried to unlock the door with a spoon.",
        "I left the groceries on the bus.",
        "I forgot how to use the toaster.",
        "I put my keys in the sugar jar.",
        "I forgot the name of my spouse briefly.",
        "I left the iron on all day.",
        "I forgot how to turn on the TV.",
        "I put my socks in the breadbox.",
        "I forgot the name of my own doctor.",

        # Tier 3: Severe (60 samples)
        "I tried to brush my teeth with hand cream this morning.",
        "I didn’t recognize my own house and insisted someone had moved my furniture.",
        "I poured laundry detergent into my coffee and drank it.",
        "I didn’t know what a fork was for while eating dinner.",
        "I kept asking who the man in the mirror was—it frightened me.",
        "I accused my daughter of stealing my shoes, though they were under my bed.",
        "I forgot how to get dressed and wore my pants on my arms.",
        "I insisted the year was 1982 and asked where my father was—he passed long ago.",
        "I walked outside at night thinking it was morning and got lost.",
        "I refused to eat because I thought the food was poisoned.",
        "I didn’t know how to unlock the bathroom door and panicked.",
        "I tried to use the TV remote as a phone and got frustrated when no one answered.",
        "I put all my clothes in the freezer because I thought it was the closet.",
        "I thought my husband was a stranger and wouldn’t let him in the house.",
        "I kept asking when I was going to school, even though I haven’t been in decades.",
        "I tried to feed the dog with a hairbrush.",
        "I didn’t recognize my own reflection and thought it was someone else.",
        "I put the toaster in the bathtub.",
        "I forgot how to speak mid-sentence and just stared blankly.",
        "I thought my son was my brother and asked about our parents.",
        "I tried to wash dishes with shampoo.",
        "I didn’t recognize my own bedroom.",
        "I tried to answer the TV when someone on-screen asked a question.",
        "I thought my medication was candy and ate too much.",
        "I forgot how to sit down and just stood confused.",
        "I tried to drink from a book.",
        "I didn’t know my own name when asked.",
        "I thought the microwave was a telephone.",
        "I tried to wear a lampshade as a hat.",
        "I forgot how to swallow and spat out my food.",
        "I thought the newspaper was a blanket.",
        "I tried to write with a banana.",
        "I didn’t recognize my own voice on a recording.",
        "I thought the cat was a baby and tried to feed it with a bottle.",
        "I forgot how to lie down and just stood by the bed.",
        "I tried to eat a bar of soap.",
        "I didn’t know what a chair was for.",
        "I thought the radio was talking directly to me.",
        "I tried to climb into the refrigerator.",
        "I forgot how to blink for a while.",
        "I thought the ceiling fan was a helicopter.",
        "I tried to flush my socks down the toilet.",
        "I didn’t recognize my own hands.",
        "I thought the shower was a car wash.",
        "I tried to comb my hair with a fork.",
        "I forgot how to breathe for a few seconds.",
        "I thought the couch was a boat.",
        "I tried to mail a sandwich.",
        "I didn’t know what a bed was for.",
        "I thought the doorbell was a fire alarm.",
        "I tried to drink from a shoe.",
        "I forgot how to nod my head.",
        "I thought the TV remote was a sandwich.",
        "I tried to wear a pot as a helmet.",
        "I didn’t recognize my own family photos.",
        "I thought the vacuum cleaner was a pet.",
        "I tried to eat a lightbulb.",
        "I forgot how to smile.",
        "I thought the window was a painting.",
        "I tried to talk to a painting.",
    ],
    "label": [
        # Tier 0: Healthy (60 samples)
        *([0] * 60),
        # Tier 1: Mild Concern (60 samples)
        *([1] * 60),
        # Tier 2: Moderate (60 samples)
        *([2] * 58),
        # Tier 3: Severe (60 samples)
        *([3] * 60),
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