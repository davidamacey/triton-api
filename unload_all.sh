#!/bin/bash
# Unload all Triton models
echo "Unloading all models..."
for model in $(curl -s -X POST http://localhost:4600/v2/repository/index | jq -r '.[].name'); do
    echo "  Unloading $model..."
    curl -s -X POST "http://localhost:4600/v2/repository/models/$model/unload" > /dev/null
done
sleep 3
echo ""
echo "READY models remaining:"
curl -s -X POST http://localhost:4600/v2/repository/index | jq -r '.[] | select(.state == "READY") | .name'
