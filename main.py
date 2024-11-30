import os
import discord
from discord.ext import commands
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# set up intents
intents = discord.Intents.default()
intents.message_content =  True

class AIFriendBot(discord.Client):
    def __init__(self):
        super().__init__(command_prefix='hey cosmic', intents=intents)

        # load pre-trained model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

        self.conversation_history = {}
        self.max_history = 10 #limit coversation context

    async def on_ready(self):
        print(f'logged in as {self.user}')

    def web_search(self, query):
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url)
            data = response.json()

            #extract first abstract result
            if data.get('AbstractText'):
                return data['AbstractText']
            elif data.get('RelatedTopics'):
                return data['RelatedTopics'][0]['Text']
            else:
                return "Sorry, no search results found."
        except Exception as e:
            return f"Search Error: {str(e)}"
        
    def generate_ai_response(self, message, user_id):
        # manage conversation history
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []

        # prepare input
        self.conversation_history[user_id].append(message)

        # trim history
        if len(self.conversation_history[user_id]) > self.max_history:
            self.conversation_history[user_id] = self.conversation_history[user_id][-self.max_history:]
        
        # generate response
        try:
            # encode conversation history
            input_ids = self.tokenizer.encode(
                " ".join(self.conversation_history[user_id]) + self.tokenizer.eos_token,
                return_tensors='pt'
            )

            # generate bot response
            chat_history_ids = self.model.generate(
                input_ids,
                max_length=1000,
                pad_token_id=self.tokenizer.eos_token_id
            )

            # decode response
            response = self.tokenizer.decode(
                chat_history_ids[:, input_ids.shape[-1]:][0],
                skip_special_tokens=True
            )
            
            return response
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
        
    async def on_message(self, message):
        # ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # respond to direct mentions
        if self.user.mentioned_in(message) or message.content.startswith('hey cosmic'):
            # remove bot mention from message
            clean_message = message.content.replace(f'<@{self.user.id}>', '').replace('hey cosmic', '').strip()

            # check if it's a search request
            if clean_message.startswith('search:'):
                search_query = clean_message[7:].strip()
                response = self.web_search(search_query)
                await message.channel.send(f"🔍 Search results for 'search_query':\n{response}")
            else:
                # generate ai reponse
                response = self.generate_ai_response(clean_message, message.author.id)
                await message.channel.send(response)

# create and run bot
def main():
    bot = AIFriendBot()
    bot.run(os.getenv('DISCORD_TOKEN'))

if __name__ == '__main__':
    main()
