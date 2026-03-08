import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
import gender_guesser.detector as gender
from sklearn.preprocessing import MinMaxScaler
import re

class LimpiezaVariables(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        horas_moodle = X['horas_moodle'].astype(str).str.lower().str.strip().str.replace(',', '.', regex=False)
        moodle_num = horas_moodle.str.extract(r'(\d+(?:\.\d+)?)', expand=False)
        X['horas_moodle'] = pd.to_numeric(moodle_num, errors='coerce')
        X['bachillerato'] = X['bachillerato'].str.lower()
        X['bachillerato'] = X['bachillerato'].replace('ccss','ciencias sociales')
        X['bachillerato'] = X['bachillerato'].replace('letras','humanidades')
        X['bachillerato'] = X['bachillerato'].replace('económico','ciencias sociales')
        X['bachillerato'] = X['bachillerato'].replace('ciencias puras', 'ciencias')
        X['bachillerato'] = X['bachillerato'].replace('científico', 'ciencias')
        X['bachillerato'] = X['bachillerato'].replace('cientifico-tecnologico', 'tecnológico')
        X['bachillerato'] = X['bachillerato'].replace('tecnológico industrial', 'tecnológico')
        return X

class ImputarHorasTrabajo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X['trabaja'] == 0, 'horas_trabajo'] = 0

        X['horas_trabajo_missing'] = np.where(X['horas_trabajo'].isna(),1,0)
        X['horas_trabajo'] = X['horas_trabajo'].fillna(23)
        
        return X
    
class RatioCredito(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['ratio_creditos_superados'] = (X['superados_a1'] / X['creditos_a1']).round(2)
   
        return X

class ImputarNota(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['nota_s1_missing'] = np.where(X['nota_s1'].isna(),1,0)
        cols_ref = ['nota_s1', 'satisfaccion', 'ratio_creditos_superados']
        df_impute = X[cols_ref]

        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df_imputed = imputer.fit_transform(df_impute)

        X['nota_s1'] = df_imputed[:, 0].round(2)
        return X

class Localidad(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['prov_estudia'] = X['id'].str.split('-', expand=True)[0]

        X['prov_estudia'] = X['prov_estudia'].replace('TO','Toledo')
        X['prov_estudia'] = X['prov_estudia'].replace('CU','Cuenca')
        X['prov_estudia'] = X['prov_estudia'].replace('CR','Ciudad Real')
        X['prov_estudia'] = X['prov_estudia'].replace('AB','Albacete')

        X['es_local'] = np.where(X['provincia'] == X['prov_estudia'], 1, 0)

        X['residencia_id'] = X['residencia_id'].fillna('0')
        X['domicilio'] = np.where(X['residencia_id'] == '0', 'Piso', 'Residencia')

        condiciones = [
            (X['es_local'] == 0) & (X['domicilio'] == 'Piso'),
            (X['es_local'] == 1) & (X['domicilio'] == 'Piso'),
            (X['es_local'] == 1) & (X['domicilio'] == 'Residencia'),
            (X['es_local'] == 0) & (X['domicilio'] == 'Residencia'),
        ]

        categorias = ['Piso_no_local', 'Piso_local', 'Residencia_local', 'Residencia_no_local']

        X['domicilio'] = np.select(condiciones, categorias,default='Sin especificar')
        
        return X

class Genero(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        d = gender.Detector()
        X['genero'] = X['nombre'].str.split().str[0].apply(lambda x: d.get_gender(x))

        X['genero'] = X['genero'].replace('mostly_male', 'male')
        X['genero'] = X['genero'].replace('mostly_female', 'female')
        X['genero'] = X['genero'].replace('andy', 'unknown')
        
        return X

class EdadEntrada(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['año_entrada'] = (X['id'].str.split('-', expand=True)[1]).astype(int)

        X['nacimiento'] = pd.to_datetime(X['nacimiento'])
        X['año_nacimiento'] = X['nacimiento'].dt.year

        X['edad_entrada'] = X['año_entrada'] - X['año_nacimiento']
        
        return X

class Participacion(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.cols_participacion = ['uso_biblioteca', 'eventos', 'tutorias']

    def fit(self, X, y):
        return self

    def transform(self, X):
        X = X.copy()
        X['participacion'] = X[self.cols_participacion].sum(axis=1) / len(self.cols_participacion)

        return X

class Convocatoria(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['convocatoria'] = np.where(X['meses_matriculado'] == 6, 'ordinaria', 'extraordinaria')
        
        return X

class GrupoTrabajo(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        dicc = (X['grupo_trabajo'].value_counts().sort_values()).to_dict()
        X['tamaño_grupo'] = X['grupo_trabajo'].map(dicc)
        
        return X

class ExtractComentario(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        serie = pd.Series(X.iloc[:, 0]).fillna("sin_comentario").astype(str)
        
        return serie

class ImputacionComentarios(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.PALABRAS_BUENAS = [
            "excelente", "sobresaliente", "destacado", "proactivo", "activo", 
            "líder", "responsable", "ayuda", "curiosidad", "mejora", "participativo", 
            "destaca", "ejemplar", "gran", "activo", "positivo", "motivado", "brillante", 
            "interés", "participación", "mejorado", "compromiso", "constante", "ajuste",
            "autonomía", "gestión", "innovador", "autodidacta", "creativo", "disciplinado",
            "colaborativo", "debe seguir así", "implicado", "destreza", "responsable",
            "iniciativa", "altas", "superior", "intento", "avance", "progreso", "resultados", 
            "actitud", "destreza", "organización", "comunicación", "autoestudio", "capacidad",
            "líder natural", "adaptable", "entusiasta", "admirable"
        ]

        self.PALABRAS_MALAS = [
            "bajo rendimiento", "plagio", "no entrega", "falta de motivación", "problemas",
            "dificultad", "ansiedad", "inasistencia", "no participa", "abandonar", "suspensos", 
            "bloqueo", "se plantea abandonar", "dudas sobre continuar", "decreciente", "problemas",
            "falta de atención", "ausencias", "desinterés", "no responde", "suspensión", 
            "desorganizado", "procrastinación", "poco esfuerzo", "falta de compromiso", "mala gestión",
            "no asiste", "dificultades", "baja participación", "inseguridad", "abandono", 
            "inmadurez", "no entrega", "problemas personales", "bajo nivel", "bajo desempeño",
            "dificultad de adaptación", "estresado", "desaparecido", "falta de enfoque", "falta de apoyo",
            "dudas sobre la titulación", "bajo compromiso", "rendimiento decreciente", "apático",
            "falta de interacción", "ausente", "dudas sobre futuro", "sin metas claras", "incapaz de superar",
            "poco interés", "falta de claridad", "resultados pobres", "estresado", "no sigue el ritmo",
            "baja motivación", "desinterés", "bajo rendimiento académico"
        ]

        self.PALABRAS_ABANDONO = [
            "abandonar", "dejar la carrera", "plantea abandonar", "dificultades personales",
            "abandono por falta de motivación", "problemas económicos", "abandono por motivos personales",
            "abandona", "abandono por dificultades", "deja todo", "se plantea dejarlo", 
            "abandono por problemas familiares", "expresa dudas sobre continuar", "abandono estudios"
        ]
    def fit(self, X, y=None):
        return self
    
    def clasificar_comentarios(self, text):
        """Clasifica los comentarios como bueno, malo o neutro, y también si mencionan abandono"""
        
        if pd.isna(text):
            return 'nulo', 0  # Si el comentario está vacío, clasificado como 'nulo' con abandono 0
        
        text = text.lower()  # Convertir el texto a minúsculas para comparación
        
        # Verificar abandono
        if any(palabra in text for palabra in self.PALABRAS_ABANDONO):
            return 'malo', 1  # Si contiene abandono, lo marcamos como malo y 1 en abandono
        
        # Clasificación positiva
        if any(palabra in text for palabra in self.PALABRAS_BUENAS):
            return 'bueno', 0  # Comentario bueno
        
        # Clasificación negativa
        if any(palabra in text for palabra in self.PALABRAS_MALAS):
            return 'malo', 0  # Comentario malo
        
        return 'neutro', 0  # Si no hay coincidencias, clasificado como neutro

    def transform(self,X):
        X['comentario_clasificado'], X['abandono_binario'] = zip(*X['comentarios'].apply(self.clasificar_comentarios))
        return X
    
class NuevasVariables(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['superados_0'] = X['superados_a1'].apply(lambda x: 1 if x == 0 else 0)
        X['mes_matriculado'] = X['meses_matriculado'].apply(lambda x: 1 if x == 1 else 0)

        X[['prov_estudia', 'año_entrada', 'num_aleat']] = X['id'].str.split('-', expand=True)
        X = X.drop(columns=['num_aleat'])

        X['prov_estudia'] = X['prov_estudia'].replace('TO','Toledo')
        X['prov_estudia'] = X['prov_estudia'].replace('CU','Cuenca')
        X['prov_estudia'] = X['prov_estudia'].replace('CR','Ciudad Real')
        X['prov_estudia'] = X['prov_estudia'].replace('AB','Albacete')

        #hazme una variable combinada con Edificio_prov_estudia.
        X['edificio'] = X['residencia_id'].str.extract(r'^(0|[A-Z0-9]+)-?')
        X['edificio_prov_estudia'] = X['edificio'] + "_" + X['prov_estudia']
        X['edificio_prov_estudia'] = X['edificio_prov_estudia'].fillna('Piso')

        X['edad_entrada_bin'] = np.where(X['edad_entrada'] <= 21, 1, 0)

        X['nota_s1_baja'] = np.where(X['nota_s1'] <= 2, 1, 0)

        X['grupo_tamaño_1'] = np.where(X['tamaño_grupo'] == 1, 1, 0)

        #creame una variable binaria para cuando las horas_trabajo sean 40.
        X['horas_trabajo_40'] = np.where(X['horas_trabajo'] == 40, 1, 0)

        return X
    
class Discretizar(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['ratio_creditos_superados_disc'] = pd.cut(X['ratio_creditos_superados'], bins=[0, 0.3, 0.6, 1.0], labels=['bajo', 'medio', 'alto'])
        X['nota_s1_disc'] = pd.cut(X['nota_s1'], bins=[0, 5, 6, 8, 10], labels=['suspenso', 'aprobado', 'notable', 'sobresaliente'])
        X['nota_acceso_disc'] = pd.cut(X['nota_acceso'], bins=[0, 7, 9, 12, 14], labels=['baja', 'media', 'alta', 'excelente'])

        X['creditos_a1'] = X['creditos_a1'].astype('string')

        X['categoria_satisfaccion'] = pd.cut(X['satisfaccion'], bins=[0, 2, 3.5, 4.5, 5], labels=['baja', 'neutro', 'satisfecho', 'muy satisfecho'])

        X['horas_moodle_disc'] = pd.qcut(X['horas_moodle'], q=4, labels=['muy baja', 'baja', 'alta', 'muy alta'])

        X['nivel_participacion'] = pd.qcut(
            X['participacion'],
            q=4,
            labels=['Muy baja', 'Baja', 'Media', 'Alta']
        )

        X['tamaño_grupo'] = X['tamaño_grupo'].astype('string')

        condiciones = [
        (X['horas_trabajo'] == 0) ,
        (X['horas_trabajo'] >= 1) & (X['horas_trabajo'] < 10),
        (X['horas_trabajo'] >= 10) & (X['horas_trabajo'] < 20),
        (X['horas_trabajo'] >= 20) & (X['horas_trabajo'] < 30),
        (X['horas_trabajo'] >= 30) ,
        ]

        categorias = ['No_trabaja', 'Pocas_horas', 'Medio_tiempo','Mucho_tiempo', 'Jornada_completa']

        X['horas_trabajo'] = np.select(condiciones, categorias,default='Sin especificar')

        bins = [0, 18, 21, 25, 35, 100]
        labels = [
            '≤18 años',
            '19–21 años',
            '22–25 años',
            '26–35 años',
            '>35 años'
        ]
        X['grupo_edad_entrada'] = pd.cut(X['edad_entrada'], bins=bins, labels=labels, include_lowest=True)

        return X

