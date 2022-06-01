import numpy as np
from flask import Flask, render_template, request
import my_tf_mod
from io import BytesIO
import matplotlib.pyplot as plt
import base64


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET','POST'])
def pred():
    image_data =request.form["file"]
    print(image_data);
    print(image_data)
    org_img, img= my_tf_mod.preprocess(image_data)

    print(img.shape)
    fruit_dict=my_tf_mod.classify_fruit(img)
    rotten=my_tf_mod.check_rotten(img)

    img_x=BytesIO()
    plt.imshow(org_img/255.0)
    plt.savefig(img_x,format='png')
    plt.close()
    img_x.seek(0)
    plot_url=base64.b64encode(img_x.getvalue()).decode('utf8')



    return render_template('Pred3.html', fruit_dict=fruit_dict, rotten=rotten, plot_url=plot_url)






if __name__=='__main__':
    app.run(debug=True)
