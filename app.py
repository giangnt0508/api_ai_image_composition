from flask import Flask, jsonify, request

app = Flask(__name__)

# Sample data
books = [
    {"id": 1, "title": "To Kill a Mockingbird", "author": "Harper Lee"},
    {"id": 2, "title": "1984", "author": "George Orwell"},
    {"id": 3, "title": "The Great Gatsby", "author": "F. Scott Fitzgerald"}
]

# Route to get all books
@app.route('/books', methods=['GET'])
def get_books():
    return jsonify(books)

# Route to get a specific book
@app.route('/books/<int:book_id>', methods=['GET'])
def get_book(book_id):
    book = next((book for book in books if book['id'] == book_id), None)
    if book:
        return jsonify(book)
    return jsonify({"error": "Book not found"}), 404

# Route to add a new book
@app.route('/books', methods=['POST'])
def add_book():
    new_book = request.json
    new_book['id'] = len(books) + 1
    books.append(new_book)
    return jsonify(new_book), 201

if __name__ == '__main__':
    app.run(debug=True)