import streamlit as st
import pandas as pd
import africastalking
import os
import sys
import requests
import google.generativeai as genai

sys.path.insert(1, './modules')
print(sys.path.insert(1, '../modules/'))

from func import generate_otp, send_sms, make_call

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))

africastalking.initialize(
    username='EMID',
    api_key = os.getenv("AT_API_KEY")
)

sms = africastalking.SMS
airtime = africastalking.Airtime


@st.dialog("Login")
def sign_in_form():
    
    phone = st.number_input('Phone number:', value=None, min_value=0, max_value=int(10e10))
    password = st.text_input("Password: ", type="password")

    st.markdown('click to recieve OTP')
    sms, call = st.columns(2)

    with sms:
        sms_btn = st.button('sms', use_container_width=True)
        if sms_btn==True:
            otp_text = generate_otp()
            send_sms(phone, otp_text)
            # get_otp_sms(phone)


    with call:
        call_btn = st.button('call', use_container_width=True)
        if call_btn==True:
            otp_text = generate_otp()
            make_call(phone)
            # get_otp_call(phone)

    otp_code = st.text_input('Enter OTP')


    submit_personal_details = st.button("🕵️ Sign in", use_container_width=True)
    if submit_personal_details==True:
        st.page_link("pgs/main.py", label="Redirecting...", icon="➡️")


sign_in_form()


