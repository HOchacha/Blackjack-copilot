from inference import get_model

model = get_model(model_id="playing-cards-ow27d/4", api_key="ojY8REpHZTPZZ0HZSh7t")

results = model.infer("videoframe_55508.png")
print(results)
