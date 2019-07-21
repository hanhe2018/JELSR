
# coding: utf-8

# In[ ]:


import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JELSR",                                     
    version="0.0.1",                                       
    author="hanhe",                                      
    author_email="hanhe@ucdavis.edu",                     
    description="JELSR_Feature_Selection",                  
    long_description=long_description,                      
    long_description_content_type="text/markdown",          
    url="https://github.com/hanhe2018/JELSR/blob/master/",            
    packages=setuptools.find_packages(),               
    classifiers=[                                         
        "Programming Language :: Python :: 3",             
        "License :: OSI Approved :: MIT License",           
        "Operating System :: OS Independent",               
    ],
)


# In[1]:


import setuptools

