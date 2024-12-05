import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.ensemble import GradientBoostingClassifier

# Configure the page
st.set_page_config(
    page_title="Klasifikasi ISPA",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        pointer-events: none;
    }
    .stApp {
        # background-color: #ffffff;
    }
    footer {visibility: hidden;}
    [data-testid="stDecoration"] {
        visibility: hidden;
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<h1 class="title">Perbandingan Metode Naive Bayes, GBM dan KNN Terhadap Pasien Penyakit ISPA Studi Kasus: Rumah Sakit Royal Taruma</h1>', unsafe_allow_html=True)
# Sidebar
st.sidebar.title("Klasifikasi ISPA")
st.sidebar.markdown("Aplikasi ini membandingkan performa dari tiga algoritma berbeda:")
st.sidebar.markdown("- Naive Bayes")
st.sidebar.markdown("- Gradient Boosting")
st.sidebar.markdown("- K-Nearest Neighbors (KNN)")
split_ratio = st.sidebar.selectbox(
    "Eksperimen",
    ("70:30", "80:20", "90:10")
)
split_randomstate = st.sidebar.selectbox(
    "Random State",
    (42, 12345, 1234, 123, 12)
)
def get_random_state(split_randomstate):
    if split_randomstate == 42:
        return 42
    elif split_randomstate == 12345:
        return 12345
    elif split_randomstate == 1234:
        return 1234
    elif split_randomstate == 123:
        return 123
    elif split_randomstate == 12:
        return 12
    
random_state = get_random_state(split_randomstate)
def get_test_size(split_ratio):
    if  split_ratio == "70:30":
        return 0.3
    elif split_ratio == "80:20":
        return 0.2
    elif split_ratio == "90:10":
        return 0.1
test_size = get_test_size(split_ratio)

# Model Parameters (optional)
st.sidebar.subheader("Parameter Model")
n_neighbors = st.sidebar.selectbox(
    "Jumlah Tetangga untuk KNN", 
    options=[3, 5, 7, 9],
    index=0 
)
learning_rate = st.sidebar.selectbox(
    "Learning Rate untuk Gradient Boosting Machine",
    options=[0.01, 0.1],
    index= 0
)
n_estimators = st.sidebar.selectbox(
    "Jumlah Estimator untuk Gradient Boosting Machine",
    options=[100, 200],
    index= 0
)
max_depth = st.sidebar.selectbox(
    "Maksimal Depth untuk Decision Tree",
    options=[3, 5],
    index= 0
)
# learning_rate = st.sidebar.slider("Learning Rate untuk Gradient Boosting", min_value=0.01, max_value=1.0, value=0.1)
# n_estimators = st.sidebar.slider("Estimators untuk Gradient Boosting", min_value=50, max_value=200, value=100)
# max_depth = st.sidebar.slider("Maximum Depth untuk Gradient Boosting", min_value=1, max_value=5, value=3)
# File upload
uploaded_file = st.file_uploader('Upload File CSV', type=['csv'])

if uploaded_file is not None and st.button("Submit"):
    try:
        # Fungsi untuk mengelompokkan usia
        def create_age_groups(age):
            if age < 1:
                return 'Bayi'
            elif 1 <= age < 5:
                return 'Balita'
            elif 5 <= age < 18:
                return 'Remaja'
            elif 18 <= age < 60:
                return 'Dewasa'
            else:
                return 'Lansia'

        # Read and display raw data
        df = pd.read_csv(uploaded_file, sep=';', decimal=',')
        st.subheader("Data Mentah")
        st.dataframe(df)
        
        # Data Preprocessing
        with st.expander("Preprocessing Data"):
            st.write("Missing Values sebelum preprocessing:")
            st.write(df.isnull().sum())
            
            # Preprocessing steps
            df['rr'] = df['rr'].fillna(df['rr'].mean())
            df['Spo2'] = df['Spo2'].replace('%', '', regex=True).astype(float)
            df['Spo2'] = df['Spo2'].fillna(df['Spo2'].mean())
            df['hr'] = df['hr'].replace(' bpm', '', regex=True).astype(float)
            df['hr'] = df['hr'].fillna(df['hr'].mean())
            df['hr'] = df['hr'].astype(int)
            df['suhu_tubuh'] = df['suhu_tubuh'].fillna(df['suhu_tubuh'].mean())
            
            # Sesak napas preprocessing
            df['sesak_napas'] = df['sesak_napas'].replace({'iya': 1, 'tidak': 0})
            df['sesak_napas'] = df['sesak_napas'].replace(' ', np.nan)
            df['sesak_napas'] = df ['sesak_napas'].fillna(df['sesak_napas'].mean())
            df['sesak_napas'] = df['sesak_napas'].astype(float)

            # Categorical encoding for age_group
            df['age_group'] = df['age'].apply(create_age_groups)  # Define age_group based on age
            df['age_group'] = df['age_group'].map({
                'Bayi': 0, 'Balita': 1, 'Remaja': 2, 'Dewasa': 3, 'Lansia': 4
            })

            # Categorical encoding
            categorical_columns = ['mual', 'pilek', 'demam', 'gender', 'batuk', 'ispa']
            for col in categorical_columns:
                df[col] = df[col].replace({'iya': 1, 'tidak': 0, 'lk': 1, 'pr': 0})
                df[col] = df[col].fillna(df[col].mean())
            
            st.write("Missing Values setelah preprocessing:")
            st.write(df.isnull().sum())
            
            st.write("Data setelah preprocessing:")
            st.dataframe(df)

        # Visualizations
        with st.expander("Visualisasi Data"):
            # Create two columns for the plots
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribusi Kasus ISPA Berdasarkan Kelompok Usia")

                # Filter data untuk kasus ISPA saja
                ispa_data = df[df['ispa'] == 1]
                
                # Hitung jumlah kasus per kategori usia
                age_group_counts = ispa_data['age_group'].value_counts().reindex([0, 1, 2, 3, 4]).fillna(0)
                
                # Plot pie chart
                fig1, ax1 = plt.subplots(figsize=(10, 8))
                plt.pie(age_group_counts, 
                        labels=['Bayi', 'Balita', 'Remaja', 'Dewasa', 'Lansia'], 
                        autopct='%1.1f%%', 
                        startangle=90, 
                        colors=['#FF9999', '#66B3FF', '#99FF99', '#FFCC99', '#9966FF'])
                plt.title('Distribusi Kasus ISPA Berdasarkan Kelompok Usia', fontsize=15)
                plt.axis('equal')
                st.pyplot(fig1)
            
            with col2:
                st.subheader("Distribusi Gender untuk Kasus ISPA")
                gender_counts = ispa_data['gender'].value_counts()
                labels = ['Perempuan' if gender == 0 else 'Laki-laki' for gender in gender_counts.index]
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                plt.pie(gender_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
                plt.title('Distribusi Gender untuk Kasus ISPA')
                plt.axis('equal')
                st.pyplot(fig2)

            def naive_bayes(X_train, y_train, X_test):
                classes = np.unique(y_train)
                priors = {c: np.mean(y_train == c) for c in classes}
                likelihoods = {}
                
                for c in classes:
                    likelihoods[c] = {}
                    X_c = X_train[y_train == c]
                    for feature in X_train.columns:
                        likelihoods[c][feature] = (X_c[feature].mean(), X_c[feature].std())
                
                def gaussian_pdf(x, mean, std):
                    exponent = np.exp(-0.5 * ((x - mean) ** 2) / (std ** 2))
                    return (1 / (std * np.sqrt(2 * np.pi))) * exponent
                
                def predict_nb(sample):
                    posteriors = {}
                    for c in classes:
                        prior = priors[c]
                        posterior = np.log(prior)
                        for feature in sample.index:
                            mean, std = likelihoods[c][feature]
                            posterior += np.log(gaussian_pdf(sample[feature], mean, std))
                        posteriors[c] = posterior
                    return max(posteriors, key=posteriors.get)

                predictions = X_test.apply(predict_nb, axis=1)
                return predictions


            def knn(X_train, y_train, X_test, n_neighbors):
                    predictions = []
                    
                    for test_sample in X_test.iterrows():
                        distances = []
                        for i, train_sample in X_train.iterrows():
                            distance = np.sqrt(np.sum((train_sample - test_sample[1]) ** 2))
                            distances.append((distance, y_train[i]))
                        
                        distances.sort(key=lambda x: x[0])
                        k_nearest = [label for _, label in distances[:n_neighbors]]
                        predictions.append(max(set(k_nearest), key=k_nearest.count))
                    
                    return predictions

            # Fungsi sigmoid
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))

            # Log loss (cross-entropy) function
            def log_loss(y_true, y_pred):
                epsilon = 1e-15
                y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
                return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

            # Fungsi untuk menemukan split terbaik pada dataset
            def find_best_split(X, y):
                best_log_loss = float('inf')
                best_feature = None
                best_threshold = None

                n_samples, n_features = X.shape
                for feature in range(n_features):
                    thresholds = np.unique(X[:, feature])
                    for threshold in thresholds:
                        left_mask = X[:, feature] <= threshold
                        right_mask = ~left_mask
                        if left_mask.sum() == 0 or right_mask.sum() == 0:
                            continue

                        # Prediksi rata-rata untuk kelompok kiri dan kanan
                        left_pred = np.mean(y[left_mask])
                        right_pred = np.mean(y[right_mask])

                        # Hitung log-loss untuk kedua kelompok
                        left_loss = log_loss(y[left_mask], np.full_like(y[left_mask], left_pred))
                        right_loss = log_loss(y[right_mask], np.full_like(y[right_mask], right_pred))

                        # Total log-loss
                        log_loss_split = (left_mask.sum() * left_loss + right_mask.sum() * right_loss) / n_samples

                        if log_loss_split < best_log_loss:
                            best_log_loss = log_loss_split
                            best_feature = feature
                            best_threshold = threshold

                return best_feature, best_threshold

            # Decision Tree Regressor untuk model pohon keputusan
            class DecisionTreeRegressor:
                def __init__(self, max_depth=max_depth):
                    self.max_depth = max_depth
                    self.tree = None

                def fit(self, X, y):
                    X = np.array(X)
                    y = np.array(y).flatten()
                    self.tree = self._build_tree(X, y, depth=0)

                def _build_tree(self, X, y, depth):
                    n_samples, n_features = X.shape
                    if depth == self.max_depth or n_samples == 0:
                        return np.mean(y)

                    feature, threshold = find_best_split(X, y)
                    if feature is None:
                        return np.mean(y)

                    left_mask = X[:, feature] <= threshold
                    right_mask = ~left_mask

                    left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
                    right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
                    return (feature, threshold, left_subtree, right_subtree)

                def _predict_sample(self, x, node):
                    if not isinstance(node, tuple):
                        return node

                    feature, threshold, left_subtree, right_subtree = node
                    if x[feature] <= threshold:
                        return self._predict_sample(x, left_subtree)
                    else:
                        return self._predict_sample(x, right_subtree)

                def predict(self, X):
                    X = np.array(X)
                    return np.array([self._predict_sample(x, self.tree) for x in X])

            # Fungsi Gradient Boosting Classifier
            def gradient_boosting_classifier(X_train, y_train, X_test, y_test, learning_rate, n_estimators):
                # Inisialisasi prediksi awal
                f_train = np.zeros(len(X_train))
                f_test = np.zeros(len(X_test))
                
                # Inisialisasi pohon-pohon
                trees = []
                for _ in range(n_estimators):
                    # Hitung residual untuk setiap estimator
                    residual = y_train - sigmoid(f_train)
                    
                    # Train Decision Tree Regressor pada residual
                    tree = DecisionTreeRegressor(max_depth=max_depth)
                    tree.fit(X_train, residual)
                    trees.append(tree)
                    
                    # Update predictions
                    f_train += learning_rate * tree.predict(X_train)
                    f_test += learning_rate * tree.predict(X_test)
                
                # Apply sigmoid for final predictions and set threshold for binary classification
                y_pred_test = sigmoid(f_test)
                y_pred_test = (y_pred_test >= 0.5).astype(int)

                return y_pred_test


        with st.expander("Model Training dan Evaluasi"):
            features = df.drop(['id', 'ispa'], axis=1)
            target = df['ispa']
            X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=split_randomstate)

            st.write(f"Data split ratio (train:test) = {1-test_size:.0%}:{test_size:.0%}")
            st.subheader("Model KNN")
            knn_predictions = knn(X_train, y_train, X_test, n_neighbors)

            st.write("Classification Report:")
            knn_report = classification_report(y_test, knn_predictions, output_dict=True)
            st.dataframe(pd.DataFrame(knn_report).transpose())

            st.write("Confusion Matrix:")
            knn_cm = confusion_matrix(y_test, knn_predictions)
            fig_knn, ax_knn = plt.subplots(figsize=(12, 6))
            sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negatif', 'Positif'],
                       yticklabels=['Negatif', 'Positif'])
            plt.title('Confusion Matrix - KNN Model')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            st.pyplot(fig_knn)

            # Train and evaluate Naive Bayes Model
            st.subheader("Model Naive Bayes")
            nb_predictions = naive_bayes(X_train, y_train, X_test)

            st.write("Classification Report:")
            nb_report = classification_report(y_test, nb_predictions, output_dict=True)
            st.dataframe(pd.DataFrame(nb_report).transpose())

            st.write("Confusion Matrix:")
            nb_cm = confusion_matrix(y_test, nb_predictions)
            fig_nb, ax_nb = plt.subplots(figsize=(12, 6))
            sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Greens',
                       xticklabels=['Negatif', 'Positif'],
                       yticklabels=['Negatif', 'Positif'])
            plt.title('Confusion Matrix - Naive Bayes Model')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            st.pyplot(fig_nb)

            st.subheader("Model Gradient Boosting")
            st.write(f"Learning Rate: {learning_rate}, N Estimators: {n_estimators}, Max Depth {max_depth}")
            # gbm = GradientBoostingClassifier()
            # gbm.fit(X_train, y_train)
            # gbm_predictions = gbm.predict(X_test)
            gb_predictions = gradient_boosting_classifier(X_train, y_train, X_test, y_test, learning_rate, n_estimators)

            st.write("Classification Report:")
            gb_report = classification_report(y_test, gb_predictions, output_dict=True)
            st.dataframe(pd.DataFrame(gb_report).transpose())

            st.write("Confusion Matrix:")
            gb_cm = confusion_matrix(y_test, gb_predictions)
            fig_gb, ax_gb = plt.subplots(figsize=(12, 6))
            sns.heatmap(gb_cm, annot=True, fmt='d', cmap='Reds',
                       xticklabels=['Negatif', 'Positif'],
                       yticklabels=['Negatif', 'Positif'])
            plt.title('Confusion Matrix - Gradient Boosting Model')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            st.pyplot(fig_gb)
        with st.expander("Data Pada Random State"):
            st.write("Data Latih (X_train):")
            st.write(X_train)  # Menampilkan data latih

            st.write("Label Latih (y_train):")
            st.write(y_train)  # Menampilkan label latih

            st.write("Data Uji (X_test):")
            st.write(X_test)  # Menampilkan data uji

            st.write("Label Uji (y_test):")
            st.write(y_test)  # Menampilkan label uji
        # Model Comparison
        with st.expander("Perbandingan Kinerja Metode Naive Bayes, GBM dan KNN Pie Chart"):
            metrics_df = pd.DataFrame({
                'KNN': {
                    'Accuracy': knn_report['accuracy'],
                    'Precision': knn_report['macro avg']['precision'],
                    'Recall': knn_report['macro avg']['recall'],
                    'F1-Score': knn_report['macro avg']['f1-score']
                },
                'Naive Bayes': {
                    'Accuracy': nb_report['accuracy'],
                    'Precision': nb_report['macro avg']['precision'],
                    'Recall': nb_report['macro avg']['recall'],
                    'F1-Score': nb_report['macro avg']['f1-score']
                },
                'Gradient Boosting': {
                    'Accuracy': gb_report['accuracy'],
                    'Precision': gb_report['macro avg']['precision'],
                    'Recall': gb_report['macro avg']['recall'],
                    'F1-Score': gb_report['macro avg']['f1-score']
                }
            }).T
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                 # Accuracy Comparison
                fig1, ax1 = plt.subplots(figsize=(8, 8))
                plt.pie(metrics_df['Accuracy'], 
                        labels=metrics_df.index, 
                        autopct='%1.1f%%', 
                        startangle=90, 
                        colors=['#FF9999', '#66B2FF', '#99FF99'])
                plt.title('Perbandingan Accuracy antar Model')
                plt.axis('equal') 
                st.pyplot(fig1)
                
                # Precision Comparison
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                plt.pie(metrics_df['Precision'], 
                        labels=metrics_df.index, 
                        autopct='%1.1f%%', 
                        startangle=90, 
                        colors=['#FF9999', '#66B2FF', '#99FF99'])
                plt.title('Perbandingan Precision antar Model')
                plt.axis('equal')  
                st.pyplot(fig2)
            
            with col2:
                # Recall Comparison
                fig3, ax3 = plt.subplots(figsize=(8, 8))
                plt.pie(metrics_df['Recall'], 
                        labels=metrics_df.index, 
                        autopct='%1.1f%%', 
                        startangle=90, 
                        colors=['#FF9999', '#66B2FF', '#99FF99'])
                plt.title('Perbandingan Recall antar Model')
                plt.axis('equal')  
                st.pyplot(fig3)
                
                # F1-Score Comparison
                fig4, ax4 = plt.subplots(figsize=(8, 8))
                plt.pie(metrics_df['F1-Score'], 
                        labels=metrics_df.index, 
                        autopct='%1.1f%%', 
                        startangle=90, 
                        colors=['#FF9999', '#66B2FF', '#99FF99'])
                plt.title('Perbandingan F1-Score antar Model')
                plt.axis('equal')  
                st.pyplot(fig4)
        with st.expander("Perbandingan Kinerja Metode Naive Bayes, GBM dan KNN Bar Chart"):
            # fig1, ax = plt.subplots(figsize=(12, 8))
            # metrics_df.plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
            # plt.title('Perbandingan Kinerja antar Model')
            # plt.xlabel('Metrics')
            # plt.ylabel('Score')
            # plt.legend(title='Model', loc='upper right')
            # plt.xticks(rotation=0)  # Rotate metric names if needed for clarity
            # st.pyplot(fig1)
            col1, col2 = st.columns(2)
            with col1:
                fig1, ax = plt.subplots(figsize=(8, 6))
                metrics_df['Accuracy'].plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
                plt.title('Perbandingan Precision antar Model')
                plt.xlabel('Model')
                plt.ylabel('Precision Score')
                plt.xticks(rotation=0)  # Keep model names horizontal for clarity
                plt.ylim(0, 1)  # Set y-axis limit to 0-1 if precision scores are between 0 and 1
                st.pyplot(fig1)
                fig2, ax = plt.subplots(figsize=(8, 6))
                metrics_df['Precision'].plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
                plt.title('Perbandingan Precision antar Model')
                plt.xlabel('Model')
                plt.ylabel('Precision Score')
                plt.xticks(rotation=0)  # Keep model names horizontal for clarity
                plt.ylim(0, 1)  # Set y-axis limit to 0-1 if precision scores are between 0 and 1
                st.pyplot(fig2)
                with col2:
                    fig3, ax = plt.subplots(figsize=(8, 6))
                    metrics_df['Recall'].plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
                    plt.title('Perbandingan Recall antar Model')
                    plt.xlabel('Model')
                    plt.ylabel('Recall Score')
                    plt.xticks(rotation=0)  
                    plt.ylim(0, 1)  
                    st.pyplot(fig3)
                    fig4, ax = plt.subplots(figsize=(8, 6))
                    metrics_df['F1-Score'].plot(kind='bar', ax=ax, color=['#FF9999', '#66B2FF', '#99FF99'])
                    plt.title('Perbandingan F1-Score antar Model')
                    plt.xlabel('Model')
                    plt.ylabel('F1-Score')
                    plt.xticks(rotation=0)  
                    plt.ylim(0, 1)  
                    st.pyplot(fig4)
            st.subheader("Kesimpulan Perbandingan")
            best_model = metrics_df['Accuracy'].idxmax()
            best_accuracy = metrics_df['Accuracy'].max()
            
            st.write(f"""
            Berdasarkan hasil perbandingan di atas:
            Model dengan accuracy tertinggi adalah **{best_model}** dengan nilai **{best_accuracy:.3f}**
            """)
    except Exception as e:
        st.error(f"Error occurred: {str(e)}")
else:
    st.info("Please upload a CSV file to begin the analysis.")