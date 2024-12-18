from flask import Flask, request, render_template
from train import recommend_and_evaluate,user_item_matrix

app = Flask(__name__)

@app.route('/')
def index():
    """
    Render the home page with the input form for recommendations.
    """
    return render_template('index.html')

@app.route('/recommend_and_evaluate', methods=['POST'])
def recommend_and_evaluate_route():
    """
    Handle the form submission, generate recommendations, and evaluate them.
    """
    try:
        user_id = int(request.form['user_id'])
        method = request.form['method']
        num_recommendations = int(request.form['num_recommendations'])
    except (ValueError, KeyError):
        return render_template('index.html', error="Invalid input. Please check your entries.")

    # Check if user_id exists in user_item_matrix
    if user_id not in user_item_matrix.index:
        return render_template('index.html', error=f"User ID {user_id} does not exist.Enter Valid User ID")

    try:
        # Generate recommendations and evaluate
        results = recommend_and_evaluate(user_id, method, num_recommendations)
    except ValueError as e:
        return render_template('index.html', error=str(e))

    # Render the results on the same page
    return render_template(
        'index.html',
        recommendations=results["recommendations"],
        evaluation_results=results["evaluation"],
        user_id=user_id,
        method=method,
        num_recommendations=num_recommendations
    )
if __name__ == '__main__':
    app.run(debug=True)
