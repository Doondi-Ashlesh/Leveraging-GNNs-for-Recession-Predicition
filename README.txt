INSTRUCTIONS:

___________________________________________________________________________
REQUIRED LIBRARIES AND INSTALLATION COMMANDS:
___________________________________________________________________________

1. numpy ------------------------------- !pip install numpy
2. pandas ------------------------------ !pip install pandas
3. matplotlib -------------------------- !pip install matplotlib
4. seaborn ----------------------------- !pip install seaborn
5. scikit-learn ------------------------ !pip install scikit-learn
6. torch (PyTorch) --------------------- !pip install torch
7. torch_geometric --------------------- !pip install torch-geometric
8. networkx ---------------------------- !pip install networkx
9. imbalanced-learn (SMOTE) ------------ !pip install imbalanced-learn
10. gradio (for UI interface) ---------- !pip install gradio
11. joblib ----------------------------- !pip install joblib
12. tqdm (optional for progress bars) -- !pip install tqdm  ← Not explicitly imported, but useful if present

NOTE: PyTorch Geometric requires compatibility with your PyTorch and CUDA versions. Use their official install selector.



___________________________________________________________________________
CODE EXECUTION
___________________________________________________________________________

NOTE: Quick Demo Interface: This is a simplified script to test predictions using our trained TGNN model via a Gradio interface.  We can simply run this interface.py to see the results (predictions) instead of executing the whole source code.

-----------------------------------
STEPS TO RUN THE Interface.py FILE:
-----------------------------------
1. Files Required in the Same Folder: 

Make sure the below listed files are all in the same folder with Interface.py. The Interface folder already have the trained and saved files in the folder for TGNN model, In case we want to test on different models, the model architecture should be defined in the code in place of TGNN model, and all the required files: new_model_we_want_to_try.pth, scalar.pkl, feature_list.pkl, graph_edges.pt, should all be available for the particular model we want to test. 

-> Interface.py

-> TGNN_SMOTE_focal_40e.pth – Model we want to test. 

-> scaler_tgnn.pkl – StandardScaler used during training for the model

-> feature_list.pkl – Feature name list for the model

-> graph_edges.pt – Contains edge_index and edge_weight for the model

--------------------------------------------------------------------------------------------------------------------

2. Activate your environment and install minimal dependencies: !pip install torch numpy gradio scikit-learn joblib

3. open Interface.py file and Run the code.

4. Click on the localhost link in the output terminal or open browser at http://127.0.0.1:7860

5. Enter a comma-separated string of feature values (excluding date and recession label, sample inputs for each class is provided in the sample_inputs_for_interface.txt file, refer to working demo for clearer execution walk through) and click on autofill or manually input features and scroll down and click on predict to get the output.

6.Output: 

Recession / No Recession
Probabilities for each class


---------------------------------------------
STEPS TO RUN THE WHOLE ORIGINAL SOURCE CODE:
---------------------------------------------

1.Ensure the following file is in your working directory: "filled_temp_dataset.csv" (input dataset) - This file is renamed to Custom_dataset in the dataset folder, make sure to change the dataset path during execution.

2.Launch Jupyter: Open GNN_pipeline.ipynb and run all cells sequentially.

"""The notebook performs:

Preprocessing of the dataset

Graph construction and modeling

Training using GCN/TGNN

Evaluation and metric plotting

And finally runs the interface block for prediction.
""""

If configured, outputs like confusion matrices, probabilities, and visualizations will appear in the notebook. You may optionally save models and the required files scaler.pkl, feature_list.pkl, graph_edges.pt for the current model you wanna save to further test them in the interface (modification to the interface code needed!!!).



--------------------------------------------------------------------------------------------------------------
NOTE: A video demo of how the interface works (No audio ) & the final Report are also included in this folder.
--------------------------------------------------------------------------------------------------------------


THE END :) 















