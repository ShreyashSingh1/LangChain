import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title of the app
st.title('Simple Streamlit App')

# Text input
name = st.text_input("Enter your name:")
st.write(f"Hello, {name}!")

# Generating some data and plotting
x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y)
st.pyplot(plt)
