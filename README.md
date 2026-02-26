Sistema de Monitoreo de Data Drift para Modelos de Machine Learning

1. Contexto del caso de negocio
En entornos reales de Machine Learning, el desempeño de un modelo no depende únicamente de su calidad durante el entrenamiento inicial, sino de la estabilidad de los datos que recibe en producción. Con el paso del tiempo, los patrones poblacionales pueden cambiar debido a factores económicos, sociales o comportamentales. Este fenómeno, conocido como Data Drift, puede degradar progresivamente la capacidad predictiva del modelo sin que exista un error evidente en el código o en la arquitectura del sistema.

En este proyecto se desarrolla un sistema completo de monitoreo de Data Drift aplicado a un modelo de riesgo crediticio. El objetivo es detectar cambios en la distribución de los datos actuales respecto a los datos históricos utilizados como referencia, permitiendo identificar variables inestables, generar alertas automáticas y sugerir acciones correctivas como el retraining del modelo.

2. Objetivo del proyecto
El objetivo principal es diseñar e implementar un sistema de monitoreo que compare datos históricos (baseline) con datos actuales (current), calcule métricas estadísticas de drift, visualice los resultados en una aplicación interactiva desarrollada en Streamlit y permita analizar la evolución del deterioro del modelo a lo largo del tiempo. El sistema debe ser capaz de detectar cambios significativos en variables numéricas y categóricas, clasificar el nivel de severidad del drift y emitir recomendaciones automáticas basadas en umbrales definidos.

3. Metodología de monitoreo
El monitoreo se basa en la comparación sistemática entre dos conjuntos de datos. El conjunto baseline representa la distribución histórica de referencia, generalmente asociada al momento de entrenamiento del modelo. El conjunto current representa los datos nuevos recibidos en producción. La diferencia estadística entre ambos permite cuantificar el nivel de inestabilidad poblacional.

Para garantizar un análisis robusto, se implementaron múltiples métricas estadísticas complementarias. Cada métrica aporta una perspectiva diferente sobre el comportamiento de las variables.

4. Métricas implementadas
Para variables numéricas se implementó el Population Stability Index (PSI), el test de Kolmogorov-Smirnov (KS) y la divergencia de Jensen-Shannon. Para variables categóricas se implementó el test de Chi-cuadrado.

El PSI mide el cambio en la distribución poblacional dividiendo las variables en intervalos (bins) y comparando proporciones entre baseline y current. Se utilizan los siguientes umbrales de interpretación: valores menores a 0.10 indican estabilidad, valores entre 0.10 y 0.25 indican drift moderado, y valores superiores a 0.25 indican drift severo.

El test de Kolmogorov-Smirnov compara distribuciones acumuladas y permite determinar si dos muestras provienen de la misma distribución. Un p-value menor a 0.05 indica diferencias estadísticamente significativas.

La divergencia de Jensen-Shannon mide la distancia probabilística entre dos distribuciones. Es una métrica simétrica y acotada, lo que la hace más estable que otras medidas como la divergencia de Kullback-Leibler.

Para variables categóricas, el test de Chi-cuadrado permite evaluar cambios en la proporción de categorías. Un p-value menor a 0.05 indica que la distribución actual difiere significativamente de la histórica.

5. Aplicación en Streamlit
Se desarrolló una aplicación interactiva utilizando Streamlit que integra el motor de monitoreo estadístico con una interfaz visual clara y ejecutiva. La aplicación muestra una tabla con todas las variables evaluadas y sus respectivas métricas, incluyendo PSI, estadístico KS, p-value de KS, divergencia Jensen-Shannon y estadístico y p-value de Chi-cuadrado.

Cada variable es clasificada automáticamente como “No Drift”, “Drift Moderado” o “Drift Severo”, utilizando un sistema visual tipo semáforo que facilita la interpretación rápida por parte de usuarios técnicos y no técnicos.

La aplicación también presenta un resumen ejecutivo que indica cuántas variables presentan drift severo o moderado y genera mensajes automáticos con recomendaciones estratégicas, como la necesidad de evaluar retraining inmediato en caso de detectar drift crítico.

6. Visualización comparativa
El sistema incluye gráficos comparativos entre la distribución histórica y la actual mediante histogramas superpuestos. Esto permite visualizar desplazamientos en la media, cambios en la dispersión o alteraciones en la forma de la distribución. Además, se presenta un gráfico horizontal de PSI por variable con líneas de referencia en los umbrales 0.10 y 0.25, facilitando la identificación de variables críticas.

7. Análisis temporal del drift
Para simular un entorno productivo real, se implementó un módulo de análisis temporal que evalúa la evolución del drift a lo largo de múltiples periodos. Se simula un incremento progresivo en la distribución de las variables y se calcula el PSI mes a mes.

Este análisis permite identificar tendencias crecientes y detectar el momento en que una variable cruza umbrales críticos. De esta forma, el sistema no solo detecta drift en un instante puntual, sino que permite observar su comportamiento dinámico en el tiempo, replicando un escenario real de monitoreo continuo.

8. Sistema de alertas y recomendaciones
El sistema genera recomendaciones automáticas basadas en la cantidad y severidad del drift detectado. Cuando múltiples variables presentan drift severo, se emite una alerta crítica recomendando retraining inmediato. En caso de drift moderado, se sugiere monitoreo reforzado. Si no se detectan cambios significativos, se indica que el modelo opera dentro de condiciones normales.

Este enfoque permite traducir métricas estadísticas en decisiones operativas concretas.

9. Hallazgos principales
Durante la simulación se observó que el drift puede crecer de forma progresiva antes de alcanzar niveles críticos. Algunas variables mostraron alta sensibilidad al cambio poblacional, mientras que otras permanecieron estables incluso ante modificaciones controladas. Esto demuestra que no todas las variables afectan el modelo con la misma intensidad y que el monitoreo debe ser específico por variable.

10. Conclusiones
Este proyecto demuestra que el monitoreo de Data Drift es un componente esencial en cualquier sistema de Machine Learning en producción. La detección temprana de cambios poblacionales permite actuar antes de que el desempeño del modelo se degrade significativamente.

La implementación combinada de PSI, KS, Jensen-Shannon y Chi-cuadrado ofrece una evaluación estadística robusta. La integración con Streamlit facilita la interpretación ejecutiva y operativa de los resultados. El análisis temporal agrega una dimensión estratégica que permite anticipar problemas futuros.

El sistema desarrollado cumple principios fundamentales de MLOps como reproducibilidad, automatización, monitoreo continuo e interpretabilidad.

11. Ejecución del proyecto
Para ejecutar el sistema, se deben instalar las dependencias, generar los datasets baseline y current, y ejecutar la aplicación Streamlit. El sistema puede adaptarse fácilmente a datos reales de producción y ejecutarse con periodicidad definida según las necesidades del negocio.

Perfecto. Continuamos desde el punto 12 en adelante, manteniendo coherencia técnica y tono profesional para cerrar el documento de manera integral.

---

12. Despliegue del modelo mediante API

Como parte del ciclo completo de MLOps, el modelo entrenado fue serializado en formato `.pkl` utilizando `joblib` y posteriormente disponibilizado como servicio mediante una API desarrollada con FastAPI. Este enfoque permite desacoplar el modelo del entorno de desarrollo y exponerlo como un servicio accesible a través de solicitudes HTTP.

El archivo `model_deploy.py` constituye el núcleo del despliegue. En él se implementa la carga del modelo entrenado desde disco, la validación estructural de los datos de entrada y la lógica de predicción. La API define un endpoint `/predict` que acepta datos en formato JSON y soporta predicción por lotes (batch prediction), permitiendo enviar múltiples registros en una sola solicitud.

El sistema valida automáticamente que las columnas recibidas coincidan con las utilizadas durante el entrenamiento, garantizando consistencia estructural y reduciendo el riesgo de errores en producción.


13. Contenerización con Docker
Para garantizar portabilidad y reproducibilidad del entorno de ejecución, la aplicación fue contenerizada utilizando Docker. La imagen construida incluye:

El código fuente del proyecto.
El modelo serializado (`best_model.pkl`).
Las dependencias necesarias definidas en `requirements-deploy.txt`.
El servidor de aplicación Uvicorn para ejecutar FastAPI.

El archivo `Dockerfile` define una imagen basada en `python:3.11-slim`, optimizada para entornos ligeros. Se establece el directorio de trabajo, se instalan dependencias y se expone el puerto 8000 para permitir el acceso al servicio.

La ejecución del contenedor se realiza mediante:

docker build -t credit-model-api .
docker run -p 8000:8000 credit-model-api

Una vez iniciado el contenedor, la documentación interactiva de la API se encuentra disponible en:

[http://localhost:8000/docs](http://localhost:8000/docs)

Desde esta interfaz es posible realizar pruebas del endpoint `/predict` y validar el funcionamiento del modelo en tiempo real.

14. Integración con principios de MLOps
El proyecto integra múltiples dimensiones del ciclo de vida de Machine Learning:

Entrenamiento y versionado del modelo.
Serialización reproducible.
Monitoreo estadístico de Data Drift.
Visualización interactiva con Streamlit.
Despliegue mediante API.
Contenerización con Docker.

Esta arquitectura permite separar claramente las responsabilidades entre entrenamiento, monitoreo y despliegue, alineándose con buenas prácticas de ingeniería de software aplicadas a ML.


15. Reproducibilidad y escalabilidad
El uso de Docker asegura que el sistema pueda ejecutarse en cualquier entorno compatible sin conflictos de dependencias. La API desarrollada puede integrarse fácilmente en pipelines de datos, sistemas externos o arquitecturas en la nube.

El diseño modular del proyecto facilita futuras extensiones como:

Implementación de versionado de modelos.
Integración con sistemas de logging y monitoreo externo.
Despliegue en servicios cloud como AWS, GCP o Azure.
Automatización de retraining basada en métricas de drift.

16. Cierre del proyecto
El sistema desarrollado no solo cumple con los requisitos académicos planteados, sino que reproduce un escenario real de producción donde el modelo es monitoreado, evaluado y desplegado como servicio independiente.

Se cubre el ciclo completo:

Desarrollo
Validación
Monitoreo
Visualización
Despliegue
Contenerización

Este enfoque refleja una implementación integral de prácticas modernas de MLOps, asegurando que el modelo no sea un artefacto estático, sino un sistema dinámico capaz de adaptarse y mantenerse estable frente a cambios poblacionales.

