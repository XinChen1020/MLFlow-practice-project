#!/bin/bash
curl -s localhost:8000/admin/train_then_roll/sklearn-model-1 \  -H 'Content-Type: application/json' \  -d '{"registered_model_name":"DiabetesRF", "serve_image": "server_sklearn-model-1"}' |jq
curl -s http://localhost:9000/invocations \  -H "Content-Type: application/json" \  -d '{        "dataframe_split": {          "columns": ["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"],          "data": [[0.03,1,0.06,0.03,0.04,0.03,0.02,0.03,0.04,0.01]]        }      }' | jq
