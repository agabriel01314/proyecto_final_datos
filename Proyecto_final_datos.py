from neo4j import GraphDatabase
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Conexión con la base de datos Neo4j
class Neo4jConnection:
    
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
            print("Driver created successfully")
        except Exception as e:
            print("Failed to create the driver:", e)
    
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, parameters=None, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

# Configurar conexión
conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="123456789")

# Verificar si el driver se creó correctamente
if conn._Neo4jConnection__driver is None:
    print("Unable to connect to the database. Please check your credentials and connection settings.")
else:
    # Obtener datos de las universidades de Neo4j
    query = """
    MATCH (uni:Universidad)
    RETURN uni.nombre AS nombre, uni.costo AS costo, uni.ranking AS ranking, 
           uni.areas AS areas, uni.ubicacion AS ubicacion
    """
    results = conn.query(query)
    
    if results is None:
        print("No results returned. Please check your query and authentication.")
    else:
        uni_df = pd.DataFrame([dict(record) for record in results])

        # Aplicar One-Hot Encoding a las columnas categóricas 
        uni_numeric = pd.get_dummies(uni_df, columns=['areas'])

        # Excluir la columna 'nombre' y las columnas no numéricas de los datos numéricos para el modelo
        uni_numeric = uni_numeric.select_dtypes(include=['int64', 'float64'])

        # Entrenar el modelo k-NN
        knn = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(uni_numeric)

        # Función para realizar las recomendaciones
        def recomendar_universidades(filtro=None, valor=None):
            if filtro and valor:
                if filtro == 'costo':
                    uni_filtrados = uni_df[uni_df['costo'] <= float(valor)]
                elif filtro == 'ranking':
                    uni_filtrados = uni_df[uni_df['ranking'] <= float(valor)]
                elif filtro == 'areas':
                    uni_filtrados = uni_df[uni_df['areas'].str.contains(valor, case=False)]
                else:
                    print("Filtro no válido.")
                    return []
            else:
                uni_filtrados = uni_df

            if uni_filtrados.empty:
                print("No se encontraron universidades con los criterios especificados.")
                return []

            uni_filtrados_numeric = pd.get_dummies(uni_filtrados, columns=['areas'])
            uni_filtrados_numeric = uni_filtrados_numeric.drop(columns=['nombre'])

            # Seleccionar solo columnas numéricas
            uni_filtrados_numeric_array = uni_filtrados_numeric.select_dtypes(include=['int64', 'float64']).values

            # Verificación de datos
            print("Shape de uni_filtrados_numeric_array:", uni_filtrados_numeric_array.shape)
            print("Contenido de uni_filtrados_numeric_array:", uni_filtrados_numeric_array)

            # Entrenar k-NN con los datos filtrados
            n_neighbors = min(5, len(uni_filtrados_numeric))
            if n_neighbors == 0:
                print("No hay suficientes universidades para recomendar.")
                return []

            knn_filtrado = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(uni_filtrados_numeric_array)
            distances, indices = knn_filtrado.kneighbors(uni_filtrados_numeric_array)
            nombres_recomendados = uni_filtrados.iloc[indices[0]]['nombre'].values
            return nombres_recomendados

        # Menú interactivo
        while True:
            print("\nMenu de Recomendaciones")
            print("1. Recomendaciones por costo")
            print("2. Recomendaciones por ranking")
            print("3. Recomendaciones por áreas")
            print("4. Salir")
            opcion = input("Selecciona una opción: ")

            if opcion == '1':
                costomax = input("Ingresa el costo máximo: ")
                recomendaciones = recomendar_universidades(filtro='costo', valor=costomax)
                print("Universidades recomendadas:", recomendaciones)

            elif opcion == '2':
                ranking = input("Ingresa el ranking: ")
                recomendaciones = recomendar_universidades(filtro='ranking', valor=ranking)
                print("Universidades recomendadas:", recomendaciones)

            elif opcion == '3':
                areas = input("Ingresa el área: ")
                recomendaciones = recomendar_universidades(filtro='areas', valor=areas)
                print("Universidades recomendadas:", recomendaciones)

            elif opcion == '4':
                print("Saliendo...")
                break
            else:
                print("Opción no válida. Inténtalo de nuevo.")

    # Cerrar conexión
    conn.close()
