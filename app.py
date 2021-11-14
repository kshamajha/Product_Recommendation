from flask import Flask, request, render_template

# import model.py file having final model details
import model

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    user_input = request.form['txtfield']

    # First user recommendation

    result, recom_products = model.recommendation(user_input)

    if result:
        # Sentiment model call
        output = model.sentiment(recom_products)
        return render_template('index.html', username=user_input, tables=[output.to_html(classes='data', index=False)])
    else:
        return render_template('index.html', message=recom_products)


if __name__ == "__main__":
    app.run(debug=True)
