1) DATA: http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

2) Commands:
python src/data/make_dataset.py --input_path ./data/raw/review_polarity/txt_sentoken/ --output_file data/processed/data.txt
python src/models/train_model.py --train_file data/processed/data.txt --output_model models/model.pkl
python src/models/predict_model.py --input_sentence 'I am good'  --model_file models/model.pkl
"# proj_sa" 
