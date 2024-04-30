#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install --upgrade protobuf


# In[5]:


# !pip install streamlit


# In[7]:


# !pip install transformers


# In[8]:


import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# In[ ]:


get_ipython().system('pip install torch torchvision torchaudio')


# In[9]:


import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def main():
    st.title("Text Generation with Phi-3-mini-4k-instruct")
    st.write("This app uses the Phi-3-mini-4k-instruct model to generate text based on user prompts.")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-3-mini-4k-instruct", 
        device_map="cuda", 
        torch_dtype="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

    # Create text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    # User input prompt
    prompt = st.text_input("Enter your prompt here:")

    # Generate button
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                # Generate response
                output = pipe(prompt, max_length=150, do_sample=False)
                generated_text = output[0]['generated_text']
                
            # Display generated text
            st.write("#### Generated Text:")
            st.write(generated_text)
        else:
            st.warning("Please enter a prompt.")

if __name__ == "__main__":
    main()


# In[2]:


import streamlit


# In[ ]:




