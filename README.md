# 🎬 Movie Recommendation System

A content-based movie recommendation system built with Machine Learning that suggests similar movies based on user selection. The system uses cosine similarity to find movies with similar features like genres, cast, crew, keywords, and overview.

## 🌟 Features

- **Content-Based Filtering**: Recommends movies based on content similarity
- **Interactive Web Interface**: Built with Streamlit for easy user interaction
- **Fast Recommendations**: Pre-computed similarity matrix for instant results
- **5000+ Movies**: Trained on TMDB dataset with comprehensive movie information
- **Top 6 Recommendations**: Returns the most similar movies for any selected title

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
  - CountVectorizer: Text vectorization
  - Cosine Similarity: Similarity measurement
- **NLTK**: Natural language processing
  - Porter Stemmer: Word stemming

## 📊 Dataset

The project uses two datasets from TMDB (The Movie Database):

- `tmdb_5000_movies.csv`: Contains movie information (genres, keywords, overview, etc.)
- `tmdb_5000_credits.csv`: Contains cast and crew information

**Total Movies**: 4,806 movies after preprocessing

## 🧠 How It Works

### 1. **Data Preprocessing**

- Merge movies and credits datasets
- Extract relevant features: genres, keywords, cast (top 3), crew (director), overview
- Handle missing values and remove duplicates
- Parse JSON-like strings to extract meaningful data

### 2. **Feature Engineering**

- Combine all features into a single 'tags' column
- Remove spaces from multi-word names (e.g., "Sam Worthington" → "SamWorthington")
- Convert all text to lowercase for uniformity
- Apply Porter Stemming to reduce words to root form (e.g., "playing" → "play")

### 3. **Vectorization**

- Use CountVectorizer to convert text to numerical vectors
- Create 5000-dimensional feature vectors
- Remove stop words (common words like "the", "is", "in")
- Generate a bag-of-words representation

### 4. **Similarity Calculation**

- Compute cosine similarity between all movie vectors
- Create a 4806 x 4806 similarity matrix
- Cosine similarity ranges from 0 (no similarity) to 1 (identical)

### 5. **Recommendation Engine**

- Find the index of the selected movie
- Retrieve similarity scores for that movie
- Sort movies by similarity in descending order
- Return top 6 most similar movies

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7 or higher
pip (Python package manager)
```

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/RifatHossaiN47/ML-Movie-Recommendation-System.git
   cd ML-Movie-Recommendation-System
   ```

2. **Install required packages**

   ```bash
   pip install streamlit pandas numpy scikit-learn nltk
   ```

3. **Run the Jupyter Notebook** (if you need to regenerate pickle files)

   - Open `Movie Recommendation.ipynb`
   - Run all cells to generate `movies_dict.pkl` and `similarity.pkl`

4. **Run the Streamlit app**

   ```bash
   python -m streamlit run app.py
   ```

5. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL manually

## 📁 Project Structure

```
movie-recommendation-system/
│
├── app.py                          # Streamlit web application
├── Movie Recommendation.ipynb      # Data processing and model training
├── movies_dict.pkl                 # Preprocessed movie data (dictionary)
├── similarity.pkl                  # Pre-computed similarity matrix
├── tmdb_5000_movies.csv           # Movie dataset
├── tmdb_5000_credits.csv          # Credits dataset
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment configuration
├── setup.sh                        # Streamlit configuration for deployment
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## 🎯 Usage

1. Launch the application using the command above
2. Select a movie from the dropdown menu
3. Click the "Recommend" button
4. View 6 similar movie recommendations instantly

## 📈 Model Performance

- **Vectorization**: 5000 features from 4,806 movies
- **Similarity Matrix**: 23,097,636 similarity scores
- **Response Time**: Instant (pre-computed similarities)
- **Accuracy**: Content-based filtering ensures thematically similar recommendations

## 🌐 Deployment

### Deploy on Streamlit Community Cloud (Recommended)

1. Push your code to GitHub
2. Visit [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Click "Deploy"

### Deploy on Heroku

1. Install Heroku CLI
2. Login to Heroku: `heroku login`
3. Create app: `heroku create your-app-name`
4. Push code: `git push heroku main`

## 🔍 Algorithm Explained

### Cosine Similarity Formula

The system uses cosine similarity to measure the angle between two movie vectors:

$$
\text{similarity} = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

Where:

- **A** and **B** are movie feature vectors
- **n** is the number of features (5000)
- Values range from 0 (completely different) to 1 (identical)

## 🎨 Features in Detail

### Content-Based Filtering

- Analyzes movie content rather than user ratings
- Works well for new users (no cold start problem)
- Provides explainable recommendations

### Feature Extraction

- **Genres**: Action, Comedy, Drama, etc.
- **Keywords**: Themes and topics (e.g., "space", "love", "revenge")
- **Cast**: Top 3 actors in leading roles
- **Crew**: Director of the movie
- **Overview**: Plot description

### Text Processing Pipeline

1. Convert to lowercase
2. Remove spaces from names
3. Apply stemming (reduce to root words)
4. Remove stop words
5. Vectorize using CountVectorizer

## 📝 Example

**Input**: Select "Avatar"

**Output**:

- Avatar
- Guardians of the Galaxy
- Aliens
- Star Wars: The Clone Wars
- Star Trek Into Darkness
- Star Trek Beyond

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Rifat Hossain**

- GitHub: [@RifatHossaiN47](https://github.com/RifatHossaiN47)

## 🙏 Acknowledgments

- TMDB for providing the movie dataset
- Streamlit for the amazing web framework
- Scikit-learn for machine learning tools

## 📞 Support

If you have any questions or issues, please open an issue on GitHub or contact the author.

---

⭐ **Star this repository if you found it helpful!**
