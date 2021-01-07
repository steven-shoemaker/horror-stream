import streamlit as st
import keras
import deploy


loaded = keras.models.load_model("text_generator_gigantic")

#===========================================#
#              Streamlit Code               #
#===========================================#
desc = "Uses a neural network trained on over *1000* horror movies to generate sometimes good, mostly non-sensical horror movie plots."

st.title('Horror Movie Generator')
st.write(desc)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    



local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

st.subheader("Enter your name:")
seed = st.text_input('')

if st.button('Generate Text'):
    generated_text = deploy.generate_text(seed, 30, loaded, 76)
    st.subheader("Your Terrible Movie:")
    st.write(generated_text)
