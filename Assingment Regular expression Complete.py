#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd


# In[117]:


#ANS1
string1='Python Exercises, PHP exercises.'

pattern1= '\s|\W+'

replace1=':'

result1=re.sub(pattern1,replace1,string1)
print(result1)


# In[5]:


#ANS2

string2='From class 12 abhishek,akansha,eliyana,ella and esha'

pattern2= '\sa.*|\se.*'

result2= re.findall(pattern2,string2)
print(result2)


# In[112]:


#ANS3
pattern3=re.compile(r'\b\w{1,4}\b')

result3=pattern3.findall('Once there was a crow in the jungle')

print(result3)


# In[113]:


#ANS4

pattern4=re.compile(r"\b\w{3,5}\b")
                    
result4=pattern4.findall('Once there was a crow in the jungle')
print(result4)


# In[101]:


#ANS5
string5='"example (.com)", "hr@fliprobo (.com)", "github (.com)", "Hello (Data Science World)", "Data (Scientist)"'

pattern5=re.compile('\s[(]|[)]')

result5=pattern5.sub('',string5)
print(result5)


# In[92]:


#ANS6

string6='"example (.com)", "hr@fliprobo (.com)", "github (.com)", "Hello (Data Science World)", "Data (Scientist)"'

pattern6=r'\([^)]*\)'

result6=re.sub(pattern6,"",string6)

print(result6)


# In[121]:


#ANS7
string7='ImportanceOfRegularExpressionsInPython'

result7=re.findall('([A-Z][^A-Z]*)',string7)
print(result7)


# In[73]:


#ANS8

str8='RegularExpression1IsAn2ImportantTopic3InPython'
pattern8='\d'

result8=re.split(pattern8,str8)
print(result8)


# In[203]:


#ANS9

string9='RegularExpression1IsAn2ImportantTopic3InPython'

pattern9='[A-Z]|\d'
result9=re.findall(pattern9,' ',string9)
print(result9)


# In[199]:


#ANS10
string10="Hello my name is Data Science and my email address is xyz@domain.com and alternate email address is xyz.abc@sdomain.domain.com. Please contact us at hr@fliprobo.com for further information."
pattern10=('\w*[@]\w*\.\w*')

result10=re.findall(pattern10,string10)
print(result10)


# In[38]:


#ANS11

string11='Hey_guys,My new mobile number is 7056131800 and mail is abhiboss73@gmail.com'
result11= re.findall(r"[a-zA-Z0-9_]*",string11)
result11


# In[150]:


#ANS12
string12="My mobile number is so easy pls note-7056131800"

pattern12=re.search("\d.*",string12)
print(pattern12)


# In[151]:


#ANS13
string13="192.168.0.1.00.12.0"

pattern13="[0]"
result13=re.sub(pattern13,'',string13)
print(result13)


# In[153]:


#ANS14
string14=" On August 15th 1947 that India was declared independent from British colonialism, and the reins of control were handed over to the leaders of the Country"

pattern14=re.findall("\w*\s\d{2}\w*\s\d{4}",string14)
print(pattern14)


# In[137]:


#ANS15
string15="The quick brown fox jumps over the lazy dog."
 
result15=re.finditer(r'fox|dog|horse',string15)

for match_obj in result15:
    print(match_obj)


# In[104]:


#ANS16
string16="The quick brown fox jumps over the lazy dog."

search16=re.search('fox',string16)
print(search16)


# In[40]:


#ANS17
string17="Python exercises, PHP exercises, C# exercises"
search17=re.search(r'exercises',string17)
print(search17)


# In[154]:


#ANS18
String18="DataScience field is good for career growth"

result18=re.search('for',String18)
print(result18)


# In[273]:


#ANS19
string19='1999-12-22'

pattern=(r'(\d{4})-(\d{1,2})-(\d{1,2})')

result19=re.sub(pattern,"\\3-\\2-\\1",string19)
print(result19)


# In[195]:


string20="Temp of varoius pateint is 98.9,99.67,101.1"

pattern="\d+\..\d|\d+\.\d"

result20=re.findall(pattern,string20)
print(result20)


# In[192]:


string21= "My maths score is 52"

pattern21='\d{2}'

result21=re.search(pattern21,string21)
print(result21)
print(result21.span())


# In[167]:


#ANS22
string22="My marks in each semester are: 947, 896, 926, 524, 734, 950, 642"
pattern='\d+'
result22=re.findall(pattern,string22)
print(result22)


# In[173]:


#ANS23
string23="RegularExpressionIsAnImportantTopicInPython"

pattern="[A-Z][a-z]*"

result23=re.findall(pattern,string23)
print(' '.join((result23)))


# In[275]:


#ANS24
string25="Hii, i am Abhishek"

pattern="[A-Z]+[a-z]+"

result25=re.findall(pattern,string25)
print(result25)


# In[285]:


#ANS25
string26="Hello hello world world"

pattern=r'\b(\w+)(?:\W+\1\b)+' 
result26=re.sub(pattern,r'\1',string26)

print(result26)


# In[5]:


#ANS26

string27="Password for your login is abhi2202"

pattern='\w*$'

result27=re.search(pattern,string27)
print(result27)


# In[14]:


#ANS27

string28='RT @kapil_kausik: #Doltiwal I mean #xyzabc is "hurt" by #Demonetization as the same has rendered USELESS <ed><U+00A0><U+00BD><ed><U+00B1><U+0089> "acquired funds" No wo'

pattern28='\s[#]\w*'

result28=re.findall(pattern28,string28)
print(result28)


# In[183]:


#ANS28
string29="@Jags123456 Bharat band on 28??<ed><U+00A0><U+00BD><ed><U+00B8><U+0082>Those who  are protesting #demonetization  are all different party leaders"
pattern29='\W?[U]?\+\w*\W'

result28=re.sub(pattern29,"",string29)
print(result28)


# In[35]:


#ANS29
string31="Ron was born on 12-09-1992 and he was admitted to school 15-12-1999"

pattern='\d{2}-\d{2}-\d{4}'
result31=re.findall(pattern,string31)
print(result31)


# In[33]:


#ANS30

string30='The following example creates an ArrayList with a capacity of 50 elements. 4 elements are then added to the ArrayList and the ArrayList is trimmed accordingly'
pattern=re.compile(r'\b\w{2,4}\b')

result30=pattern.sub(' ',string30)
print(result30)


# In[ ]:





# In[ ]:




