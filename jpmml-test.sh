#!/usr/bin/env bash

java -cp pmml-evaluator-example-executable-1.0.0-SNAPSHOT.jar \
        org.jpmml.evaluator.example.EvaluationExample \
        --model model.pmml \
        --input jpmml-test-input.csv \
        --output output.csv

cat output.csv
