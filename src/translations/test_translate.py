from src.translations.translate import (load_model, translate)

if __name__ == "__main__":
    query =  "Protection des données et droit au respect de la vie  privée (article 8 de la Convention), Opérations sur des données susceptibles de porter atteinte au  droit au respect de la vie privée, Collecte des données à caractère personnel, Collecte de données à caractère personnel dans le contexte de la santé"
    model, tokenizer = load_model()
    translating = translate(query=query, model=model, tokenizer=tokenizer)
    
    print(translating)