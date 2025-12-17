from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz
from sklearn.model_selection import train_test_split
import graphviz

# 1. Cargar la base de datos
iris = load_iris()
X, y = iris.data, iris.target

# 2. entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Entrenar modelo (profundidad limitada)
model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X_train, y_train)

# 4. Reglas del árbol
rules = export_text(model, feature_names=iris.feature_names)
print(" Reglas del Arbol (Profundidad limitada) ")
print(rules)

# 5. Precisión
accuracy = model.score(X_test, y_test)
print("Precision prueba:", accuracy)

# 6. Modelo sin límite de profundidad
model_full = DecisionTreeClassifier(random_state=42)
model_full.fit(X_train, y_train)

rules_full = export_text(model_full, feature_names=iris.feature_names)
print("\n Reglas del Árbol (sin limite)")
print(rules_full)

accuracy_full = model_full.score(X_test, y_test)
print("Precision completa:", accuracy_full)

# 7. árbol gráfico en arbol_iris.png
dot_data = export_graphviz(
    model,
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
graph.format = "png"
graph.render("arbol_iris", cleanup=True)
