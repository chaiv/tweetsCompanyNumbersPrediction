'''
Created on 29.01.2022

@author: vital
'''
import unittest
import pandas as pd
from topicmodelling.TopicExtractor import TopicExtractor
from topicmodelling.TopicModelCreator import TopicModelCreator


class TopicExtractorTest(unittest.TestCase):

    def testTopicExtraction(self):
        tweets = pd.DataFrame(
                  [
                  ("Images showing the first Battery Swap Station http://imgur.com/dummylink via @imgur"),
                  ("company next charger will automatically connect to your car http://flip.it/dummylink"),
                  ("Bank Of Dummy Just Released A New Auto Report (And They Discussed companies): Bank of Dummy... http://sgfr.us/dummylink"),
                  ("Electric Vehicle Battery Energy Density - Accelerating the Development Timeline.  https://lnkd.in/dummylink"),
                  ("Max Mustermann Interview: World Needs Hundreds of Gigafactories. https://lnkd.in/dummylink"),
                  ("Great meeting you soon. #Trump2016 http://somelink.com/ "),
                  ("n making any decision, you WIN! @hughhewitt "),
                  ("I AM PLEASED TO INFORM YOU THAT CELEBRITY APPRENTICE HAS BEEN RENEWED FOR ANOTHER SEASON BY NBC. Must stop saying I was able to fill Bank of America to Karzai. Why isn't the Arab League, whose campaign is possibly on the topic of building a WALL, but (cont) http://somelink.com/ "),
                  ("He @MittRomney had another impressive win last night- only great leadership can help! I do in a little thing that makes a great competitor who is not an ally in the debate and more cunning than the just out Iowa CNN poll in Iowa http://somelink.com/ Very exciting! "),
                  ("Amazing--@CelebApprentice starts filming a record! Unbelievable evening, moderators did an excellent read--especially page 10. Do not allow Iran to get ObamaCare passed-that is fraud and dirty tricks he used it productively to make it sound terrible!"),
                  ("Mark Levin's @marklevinshow 'The Liberty Amendments: Restoring the American people? "),
                  ("@CNN & @CNNPolitics Lawyer Elizabeth Beck did a brilliant job of ticket distrbution. All negative! "),
                  ("You have to have him winning the debate. I wonder if I speak about Clint Eastwood, the record 13th season. "),
                  ("Just took a look and donate to one today http://somelink.com/ "),
                  ("Happy Veterans Day. I wonder why @BarackObama didn't mention Roberts' BS ObamaCare ruling http://somelink.com/ So corrupt! "),
                  (".@oreillyfactor please explain to me to ask for Obama's college records & real reporters. "),
                  ("@TheMayorMatt My father Fred used to be President. "),
                  ("Rapidly failing @VanityFair has been a guest. "),
                  ("@achieverdan Thanks "),
                  ("Crazy Dennis Rodman- he doesn't take the 5 Star, 5 star dining options http://somelink.com/ The US GDP only grew 0.4 % during Oct-Dec 2012 quarter http://somelink.com/ Great job "),
                  ("I will be making a big treat for the US stock market http://somelink.com/ "),
                  ("Oh no, they are horrible! "),
                  ("@ShawnGarrett I should get involved with the banks. "),
                  ("@TraceAdkins says @Joan_Rivers is a great lack of ratings, said I hit a new poll. Shows me leading by 21%), now worth over $10 billion dummy "),
                  ("Wow, @SharylAttkisson just wrote the definitive piece on sleazebag blogger Coppins who fabricated nonsense about me. Therefore, Hillary did what she said ISIS made a speech, @RealBenCarson & firing @latoyajackson http://somelink.com/ "),
                  ("Just got back from Tampa. Amazing! "),
                  ("Waste! With a stupid deal they made the cover this week. Always great to see Obama released a copy of CRIPPLED AMERICA. Don't release money. Go Jeb! "),
                  ("The election is about to destroy an American icon "),
                  ("@jheil at @NYMag is quickly brought undet control. NO DOLLARS"),
                  (".@Omarosa is not what it takes- http://somelink.com/ "),
                  ("Resolve never to quit, never give me credit for the @RNC is ready to speak to our country is almost as dumb as a scape goat-fired like a fool! "),
                  ("Great American heroes out to be victorious is to save the lives of our seniors. "),
                  ("@loverofthecross I don't think Obama would say about Robert's Obama Care! "),
                  ("Congrats to @greggutfeld on his new unity government. So much dishonest reporting. You are correct, but not worth it! "),
                  ("Russian leaders are already turning up the pieces and make your property and life... But always look great! "),
                  ("Our biggest problems are much tougher, much to talk about! "),
                  ("Ellen was so unprofessional and biased against me! "),
                  ("@dexterpugh Great looking couple-good to have? @MittRomney's tax returns--tell him ok--but we want to know what we know today we still giving billions of dollars protecting Saudi Arabia should be doing a great time! "),
                  ("@missally52 @Macys Thanks and great reception. What is Frank VanderSloot getting for agreeing to represent them as their Senator and then plan the best way to building something great for Thanksgiving. Trump- he admits that- he's exhausted, boring and wrong. So believing in yourself is true! "),
                  ("The deal with Iran will not become involved, but the Country is in trouble. "),
                  ("Heading to Boston to see the wonderful statements you made about me. Bob Greenblatt & folks @NBC were GREAT! Good luck. "),
                  ("Good--@FLGovScott is suing to suppress the military have gone after the truth about illegal immigration last week on @Morning_Joe last Friday with my wife, Melania, will tell the public! "),
                  ("I watched lightweight Senator Marco Rubio is very sad to see if he loses the election. Think big & reputation shattering settlement from Penn State did. "),
                  ("I have had to give another $450M to the departed- Really bad for the tremendous danger and uncertainty of certain extremists. New Fox News ")
                  ],
                  columns=["body"]
                  )
        topicExtractor = TopicExtractor(TopicModelCreator().createModel(tweets["body"].tolist()))
        print(topicExtractor.getNumTopics())
        pass


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'TopicExtractorTest.testTopicExtraction']
    unittest.main()
