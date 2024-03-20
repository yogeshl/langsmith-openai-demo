from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true" # this flag enables tracking.
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")


prompt=ChatPromptTemplate.from_messages(
    [
        ("system", "You are an helpful assitant. Please response to the user request only based on the given context"),
        ("user","Question:{question}\nContext:{context}")
    ]
)

model=ChatOpenAI(model="gpt-3.5-turbo")
output_parser=StrOutputParser()
chain=prompt|model|output_parser

question="Can you summarize the content?"
context="""
Can India Become the Next Global Superpower?
The country has become the world’s most populous, but there are doubts about whether that title heralds a growth in wealth and influence.
2023-04-27T06:00:12-04:00
This transcript was created using speech recognition software. While it has been reviewed by human transcribers, it may contain errors. Please review the episode audio before quoting from this transcript and email transcripts@nytimes.com with any questions.

Sabrina Tavernise
Hey, everybody. It’s Sabrina. Before we get started today, I just wanted to let you know that our colleagues over at The Run-Up have started releasing their second season. The show is hosted by Astead Herndon. And every week, it dives deeply into questions about American politics.

Like this week’s show, it’s a two-parter featuring back-to-back interviews, first with MyPillow CEO and election denier Mike Lindell, and then with RNC Chairwoman Ronna McDaniel. Those conversations give us a look inside the two opposing wings of the Republican Party as they jostle for control going into next year’s elections. So that’s The Run-Up.

They do a great job. They make a great show. Find them wherever you listen to us and subscribe. OK, here’s today’s show.

From The New York Times, I’m Sabrina Tavernise. And this is The Daily.

Speaker 1
The title of world’s most populous country is expected to change later this month.

Sabrina Tavernise
For the first time in at least three centuries, China is no longer the world’s most populous country.

Speaker 1
India is, in fact, on the brink of hitting a breathtaking milestone.

Sabrina Tavernise
This month, India overtook it.

Speaker 1
Alongside its population of 1.4 billion, India’s geopolitical and economic footprint is also growing.

Sabrina Tavernise
Today, Alex Travelli, the Times’s South Asia business correspondent, explains what that new status means for India’s future and whether India will be able to use it as China once did to become an economic superpower.

Speaker 2
So what does this mean when it comes to the influence of India on the global economic / I mean, could it surpass China?

Sabrina Tavernise
It’s Thursday, April 27.

So Alex, you’ve been covering the economy in India for many years now. And I really wanted to talk to you because something kind of remarkable has happened, which is that India has overtaken China as the world’s most populous country. According to the UN, there are upwards of 1.4 billion people in each country with India now a ahead. And so all of this kind of begs the question, will India be the next China?

Alex Travelli
That is really the question, Sabrina, that is being asked by so many people. I think more outside of India and China than inside of them, but we’re feeling the pressure here too. Will India beat China? Well, they have this incredible thing in common in terms of their population. They’re next door neighbors.

They both have ancient histories. And a lot of these broad strokes, they are much more alike than any other country in the world is like either. But as anyone who’s visited these places know, they’re utterly different. They’re almost literally worlds apart.

But there’s a kind of motivation here to seeing India become China or become like China in some ways. And that’s because of the meaning that China has taken on for the rest of the world since 1990. Since 1990, China has turned its enormous population, the sort of scale that represents as a country, as a national economy, to its advantage in a way that was pretty much unimaginable before.

They turned to all of these people into low-cost labor to transform China into becoming the factory of the world. And in so doing, there’s been, by almost any measure, the greatest production of mass wealth over these past 30, 40 years from the Chinese economy that has ever been measured anywhere in the history of the world. The number of people who were deeply impoverished in the China of 1990, people who are now enjoying middle-income lives measured by a global standard, we’re talking about hundreds of millions of people.

And that’s not only a terrific reduction in poverty such as India should hope to achieve for itself, India still being barely a lower middle-income country, this is also what made China an indispensable part of the world economy. And there are various factors around the world right now that have a lot of countries and companies rooting for India to displace China as a location for production, as a new consumer class, as a new geopolitical force between the Ukraine war, between COVID responses, between tensions ramping up over the tech industry in Taiwan. There is a lot of hope in some quarters certainly that India might become the next China and all of these big world-shaking ways.

Sabrina Tavernise
OK, so what’s the answer? Will India become the next China? Like will they be able to leverage their huge population in the way that China did?

Alex Travelli
OK, I don’t have a crystal ball, but there are big, big differences between India and China. And some of these differences, three of them I’m thinking of in particular, make us question whether India could ever become something like China, whether there is any other China but China. And for a start, India’s economy is just not shaped at all the way that China’s is, that China’s was in 1990. In fact, it’s not even shaped the way that industrializing economies that we’ve seen in the past before China’s astonishing success were structured.

Sabrina Tavernise
So tell me more about that.

Alex Travelli
Well, at the simplest level, there are not enough good jobs in India. That’s not to say Indians are sitting around without jobs or unemployed in great numbers, but they don’t have the kind of jobs that you’d need to catch up with China, to catch up with the kind of growth rates that India wants. Now, you’ve got an unusually — most people would say unhealthily large reliance on the services part of the economy. Then you have this traditionally very large part of the economy devoted to agriculture.

And then you’ve got the industrial or manufacturing sector, but that is relatively small. And it’s not been growing. It’s been stuck around 14 percent, 15 percent of the economy for a long time. China’s, by contrast, is almost twice as high, and it creates a very, very different sort of economic dynamic.

So this relatively small proportion of manufacturing is a problem for India. And the reason is that manufacturing makes jobs, the kind of jobs that bring broader, base of income earning which means better consumer spending down the line. It also tends to mean exports, which is great for the currency and India’s trade balances.

There are very few countries in the world that have managed to swing quickly from being poor agrarian societies to being middle-income industrial powers even high income. And they pretty much all went this route with export-driven manufacturing taking the lead. Now, India frustratingly has kept that at the low constant rate, despite a lot of efforts from a lot of different governments to hike up the mix of manufacturing. And it’s especially frustrating for India at this moment, I would offer, because we are talking about the vast number of young people entering the workforce.

Well it’s nominally bigger than China in terms of population. But actually, the graph, the sort of curve of the population by age makes India look even more attractive because it has a huge number of people in their 20s and in their 30s, somewhat fewer in their 40s. And there are a lot of young people who are about to enter the workforce too.

China, by contrast, mainly because of that one-child policy has just gone into population decline. And its population is going to be getting older. India has workers, neither does it have too many children nor does it have too many retirees. You’ve really got a population profile that’s perfectly aged to enter a bunch of great jobs, and you don’t have enough available.

Sabrina Tavernise
OK, so the problem is not enough jobs, and in particular, not enough manufacturing jobs even though they have plenty of people to fill those kind of jobs. That’s the problem right now in India. So why then is it that India doesn’t have a big manufacturing sector if they have the workers to fill it?
"""

print(chain.invoke({"question":question,"context":context}))
