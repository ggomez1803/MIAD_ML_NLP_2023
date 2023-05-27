from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from Despliegue_proyecto2 import predict_genre

app = Flask(__name__)
api = Api(app, version='1.0',
          title= 'Predictor de géneros de películas',
          description= 'API para predecir el género de una película basado en su descripción con un modelo de Suppot Vector Classifier')

ns = api.namespace('predict', description='Predicción de géneros de películas')

parser = api.parser()

parser.add_argument('Title', type=int, required=True, help='Título de la película', location='args')
parser.add_argument('plot', type=int, required=True, help='Sinápsis de la película', location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class CarPricingApi(Resource):
    
        @api.doc(parser=parser)
        @api.marshal_with(resource_fields)
        def get(self):
            args = parser.parse_args()
    
            title = args['Title']
            plot = args['pllot']
    
            return {
            "result": predict_genre(title, plot)
            }, 200
        
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8080)
    
        