# Импортираме необходимите библиотеки
import json
import sqlite3

import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics.pairwise import cosine_similarity


SAVE_ALL_BOOKS = True  # Ако е True, записва всички книги в SQLite


# Зареждаме JSON данните и ги преобразуваме в pandas DataFrame
def load_books(file_path):
    """
    Зарежда данни за книги от JSON файл и ги преобразува в pandas DataFrame.

    Параметри:
    - file_path (str): Път към JSON файла със данни за книги.

    Връща:
    - pandas DataFrame: Съдържа информация за всички книги.

    Пример:
    >>> books = load_books("data.json")
    >>> books.head()
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    books = pd.DataFrame.from_dict(data, orient="index")
    books.reset_index(drop=True, inplace=True)

    return books


# Изчисляваме матрица на сходства между книгите
def compute_similarity(books):
    """
    Изчислява матрица на сходства между книгите, използвайки TF-IDF векторизация.

    Параметри:
    - books (pandas DataFrame): DataFrame, съдържащ колона "Text" с текстовото съдържание на книгите.

    Връща:
    - numpy.ndarray: Матрица на сходства (cosine similarity) между книгите.

    Пример:
    >>> similarity_matrix = compute_similarity(books)
    >>> similarity_matrix[0]  # Показва сходствата на първата книга
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(books["Text"].fillna(""))  # Запълваме NaN стойностите с празен низ
    similarity_matrix = cosine_similarity(tfidf_matrix)  # Изчисляваме сходството между книгите
    return similarity_matrix


# Функция, която предлага препоръчани книги, базирани на търсено заглавие
def get_recommendations(title, books, similarity_matrix, top_n=5):
    """
    Препоръчва книги, подобни на даденото заглавие, като изключва самата търсена книга.

    Параметри:
    - title (str): Заглавие на търсената книга.
    - books (pandas DataFrame): DataFrame със списък на книгите.
    - similarity_matrix (numpy.ndarray): Матрица на сходства между книгите.
    - top_n (int): Брой препоръчани книги (по подразбиране 5).

    Връща:
    - pandas DataFrame: DataFrame с препоръчани книги и автори, изключвайки търсената книга.
    """
    # Намираме индекса на търсената книга
    book_index = books[books["Title"].str.lower() == title.lower()].index
    if book_index.empty:
        return pd.DataFrame(columns=["Id", "Title", "Author", "Form"])

    book_index = book_index[0]
    scores = list(enumerate(similarity_matrix[int(book_index)]))

    # Изключваме търсената книга от препоръките
    scores = [score for score in scores if score[0] != book_index]

    # Сортираме по сходство и избираме топ N
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]

    # Извличаме Id, заглавията и авторите на препоръчаните книги
    recommendations = pd.DataFrame([
        {
            "Id": books.iloc[i[0]]["Id"],
            "Title": books.iloc[i[0]]["Title"],
            "Author": books.iloc[i[0]]["Author"],
            "Form": books.iloc[i[0]]["Form"],
        }
        for i in scores
    ])

    # Задаваме Id като индекс на DataFrame-а
    recommendations.set_index("Id", inplace=True)

    return recommendations


# Функция за изчисляване на precision, recall и F1-score
def evaluate_recommendations(books, similarity_matrix):
    """
    Оценява препоръките чрез изчисляване на precision, recall и F1-score.

    Параметри:
    - books (pandas DataFrame): DataFrame със списък на книгите.
    - similarity_matrix (numpy.ndarray): Матрица на сходства между книгите.

    Връща:
    - tuple: Съдържа precision, recall и F1-score.
    """
    y_true, y_pred = [], []
    for i in range(len(books)):
        relevant_genres = set(books.iloc[i]["Genres"].split())
        recommendations = get_recommendations(books.iloc[i]["Title"], books, similarity_matrix)
        
        for rec in recommendations["Title"]:
            rec_genres = set(books[books["Title"] == rec]["Genres"].values[0].split())
            y_true.append(1 if relevant_genres & rec_genres else 0)  # Ако има съвпадение в жанровете, е релевантно
            y_pred.append(1)  # Предполага се, че препоръката е релевантна (поне за един жанр)

    # Изчисляване на метриките
    precision = precision_score(y_true, y_pred, zero_division=1)
    recall = recall_score(y_true, y_pred, zero_division=1)
    f1 = f1_score(y_true, y_pred, zero_division=1)

    return precision, recall, f1


# Функция за записване на резултатите в база данни
def save_results_to_sqlite(search_title, recommendations, books, db_path="database/books_recommendations.db"):
    """
    Записва резултатите от препоръките и информацията за книгите в SQLite база данни.

    Параметри:
    - search_title (str): Търсена книга.
    - recommendations (list): Списък с препоръки (заглавия на книги).
    - books (pandas DataFrame): DataFrame със списък на книгите.
    - db_path (str): Път към SQLite базата данни.
    """
    try:
        # Използване на контекстен мениджър за връзка с базата данни
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            # Създаване на таблица за препоръки (ако не съществува)
            cursor.execute(""" 
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    searched_title TEXT,
                    recommended_titles TEXT,
                    search_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Вмъкване на данни за препоръките
            cursor.execute(""" 
                INSERT INTO recommendations (searched_title, recommended_titles)
                VALUES (?, ?)
            """, (search_title, ", ".join(recommendations)))

            # Записване на всички книги, ако SAVE_ALL_BOOKS е True
            if SAVE_ALL_BOOKS:
                cursor.execute(""" 
                    CREATE TABLE IF NOT EXISTS books (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT,
                        author TEXT,
                        genres TEXT
                    )
                """)
                # Използване на executemany за оптимизация
                books_data = [(book["Title"], book["Author"], book["Genres"]) for _, book in books.iterrows()]
                cursor.executemany(""" 
                    INSERT OR IGNORE INTO books (title, author, genres)
                    VALUES (?, ?, ?)
                """, books_data)

            conn.commit()  # Потвърждаваме промените
        print("\nРезултатите са записани в базата данни.")

    except sqlite3.Error as e:
        print(f"\nГрешка при запис в базата данни: {e}")


# Основната функция, която зарежда книги, изчислява сходства и предоставя препоръки на потребителя.
def main():
    """
    Основната функция, която зарежда книги, изчислява сходства и предоставя препоръки на потребителя.
    """
    file_path = "./archive/data.json"  # Път до вашия JSON файл
    books = load_books(file_path)
    similarity_matrix = compute_similarity(books)

    user_input = input("Въведете заглавие на прочетена книга: ")
    print("Търсене на препоръки...")
    recommendations = get_recommendations(user_input, books, similarity_matrix)

    if not recommendations.empty:
        print("Препоръчани книги:")

        display(recommendations)  # Записване на препоръчаните книги в табличен вид
    else:
        print("Няма намерени препоръки.")

    # precision, recall, f1 = evaluate_recommendations(books, similarity_matrix)
    # print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}")

    save_results_to_sqlite(user_input, recommendations["Title"].tolist(), books)

if __name__ == "__main__":
    main()
