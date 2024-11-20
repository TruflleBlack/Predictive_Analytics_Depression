import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Membaca data dari CSV
df = pd.read_csv('Depression.csv')

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Depresi Interaktif",
    page_icon="🧠",
    layout="wide"
)

# Header gambar
st.image("header_image.jpg", use_container_width=True)


# Menu navigasi
menu = st.sidebar.radio("Pilih Menu:", ["📊 Total Statistik", "📈 Analisis Visual", "🤖 Prediksi Tingkat Depresi"])

if menu == "📊 Total Statistik":
    st.title("📊 Total Statistik")
    total_depressed = df['depressed'].sum()
    total_not_depressed = df['depressed'].count() - total_depressed
    total_married = df['Married'].value_counts().get(1, 0)
    total_not_married = df['Married'].value_counts().get(0, 0)

    # Menampilkan metrik
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🧠 Total Depresi", f"{total_depressed:,}")
    col2.metric("😊 Total Tidak Depresi", f"{total_not_depressed:,}")
    col3.metric("💍 Total Menikah", f"{total_married:,}")
    col4.metric("💔 Total Belum Menikah", f"{total_not_married:,}")

elif menu == "📈 Analisis Visual":
    st.title("📈 Analisis Visual")
    # Korelasi antar variabel
    st.subheader("Korelasi Antar Variabel")
    try:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr = numeric_df.corr()
            fig_corr, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig_corr)
        else:
            st.warning("Tidak ada data numerik untuk menghitung korelasi.")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghitung korelasi: {e}")

    # Distribusi Usia
    st.subheader("Distribusi Usia")
    fig_age = px.histogram(
        df,
        x='Age',
        title='Distribusi Usia',
        color='depressed',
        color_discrete_map={0: '#3b9dd4', 1: '#d43b3b'},
        labels={'depressed': 'Tingkat Depresi'}
    )
    st.plotly_chart(fig_age, use_container_width=True)

    # Scatterplot interaktif
    if 'depressed' in df.columns:
        selected_var = st.selectbox("Pilih Variabel untuk Scatterplot:", df.columns)
        fig_scatter = px.scatter(
            df,
            x=selected_var,
            y='depressed',
            color='depressed',
            title=f"Scatterplot: {selected_var} vs Tingkat Depresi",
            color_discrete_map={0: '#3b9dd4', 1: '#d43b3b'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.warning("Kolom 'depressed' tidak ditemukan dalam dataset.")

elif menu == "🤖 Prediksi Tingkat Depresi":
    st.title("🤖 Prediksi Tingkat Depresi")

    # Preprocessing: Konversi nilai string menjadi numerik jika diperlukan
    def preprocess_data(df, features):
        df_preprocessed = df.copy()
        for col in features:
            if df_preprocessed[col].dtype == 'object':
                try:
                    # Gunakan raw string untuk pola regex
                    df_preprocessed[col] = df_preprocessed[col].str.extract(r'(\d+)').astype(float)
                except:
                # One-hot encoding untuk kolom kategorikal
                    df_preprocessed = pd.get_dummies(df_preprocessed, columns=[col], drop_first=True)
        return df_preprocessed


    # Pilih variabel input
    features = st.multiselect("Pilih Variabel Input:", df.columns, default=["Age", "Married"])
    if features:
        try:
            # Preprocessing
            df_cleaned = preprocess_data(df, features)
            X = df_cleaned[features]
            y = df_cleaned['depressed']
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model Logistic Regression
            model = LogisticRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Evaluasi
            accuracy = accuracy_score(y_test, y_pred)
            st.write(f"**Akurasi Model:** {accuracy:.2%}")
            
            # Input data pengguna
            st.write("Masukkan data untuk prediksi:")
            user_input = {feature: st.number_input(f"Masukkan nilai {feature}:") for feature in features}
            if st.button("Prediksi"):
                pred_result = model.predict([list(user_input.values())])[0]
                st.success(f"Hasil Prediksi: {'Depresi' if pred_result == 1 else 'Tidak Depresi'}")
        except Exception as e:
            st.error(f"Error: {e}")
