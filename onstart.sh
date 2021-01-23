#!/bin/bash
# This file is run on instance start. Output in ./onstart.log
export DEBIAN_FRONTEND=noninteractive | apt-get -q -y install git
git clone https://github.com/uberkinder/TA-Transformer.git
