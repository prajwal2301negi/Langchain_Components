from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('dl-curriculum.pdf')

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)



# 23
# CampusXDeepLearningCurriculum
# A.ArtificialNeuralNetworkandhowtoimprovethem
# 1.BiologicalInspiration
# ● Understandingtheneuronstructure● Synapsesandsignal transmission● Howbiological conceptstranslatetoartificial neurons
# 2.HistoryofNeuralNetworks
# ● Earlymodels(Perceptron)● BackpropagationandMLPs● The"AI Winter" andresurgenceof neural networks● Emergenceof deeplearning
# 3.PerceptronandMultilayerPerceptrons(MLP)
# ● Single-layer perceptronlimitations● XORproblemandtheneedfor hiddenlayers● MLParchitecture
# 4. LayersandTheirFunctions
# ● InputLayer○ Acceptinginput data● HiddenLayers○ Featureextraction● OutputLayer○ Producingfinal predictions
# 5.ActivationFunctions
# {'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': 'dl-curriculum.pdf', 'total_pages': 23, 'page': 1, 'page_label': '2'}