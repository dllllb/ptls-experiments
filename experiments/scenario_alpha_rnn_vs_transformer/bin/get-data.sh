#!/usr/bin/env bash

curl -OL https://storage.yandexcloud.net/ds-ods/files/materials/02464a6f/data_for_competition.zip
unzip data_for_competition.zip
mv data_for_competition data
mv data_for_competition.zip data/
