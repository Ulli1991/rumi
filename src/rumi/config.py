"""Typed cpp-nbody configuration table with explicit compatibility validation."""

from dataclasses import dataclass
import json
import math
from pathlib import Path


SPECS=json.loads(Path(__file__).with_name('upstream_params.json').read_text())


def parse_value(key,value):
    if key not in SPECS:raise ValueError(f"unknown cpp-nbody parameter {key}")
    spec=SPECS[key]
    if not isinstance(value,str):value=str(value)
    kind=spec['type']
    if kind=='B':
        if value.lower() in ('true','1','yes'):return True
        if value.lower() in ('false','0','no'):return False
        raise ValueError(f"{key}: expected bool")
    if kind=='I':return int(value)
    if kind=='D':
        result=float(value)
        if not math.isfinite(result):raise ValueError(f"{key}: non-finite number")
        return result
    choices=spec['choices'].split('|') if spec['choices'] else []
    if choices and value not in choices:
        if '<number>' not in choices:raise ValueError(f"{key}: expected {choices}")
        float(value)
    return value


class Config:
    def __init__(self,**overrides):
        self.values={key:parse_value(key,s['default']) for key,s in SPECS.items()}
        for key,value in overrides.items():self.values[key]=parse_value(key,value)

    def __getattr__(self,key):
        values=self.__dict__.get('values',{})
        if key in values:return values[key]
        raise AttributeError(key)

    def __getitem__(self,key):return self.values[key]

    @classmethod
    def read(cls,path):
        values={}
        for number,line in enumerate(Path(path).read_text().splitlines(),1):
            line=line.split('#',1)[0].strip()
            if not line:continue
            if '=' not in line:raise ValueError(f"line {number}: expected KEY = value")
            key,value=(s.strip() for s in line.split('=',1))
            if key in values:raise ValueError(f"line {number}: duplicate {key}")
            values[key]=value
        return cls(**values)

    def write(self,path):
        Path(path).write_text('\n'.join(f'{key} = {str(value).lower() if isinstance(value,bool) else value}' for key,value in self.values.items())+'\n')

    def changed_from_defaults(self):
        return {k:v for k,v in self.values.items() if v!=parse_value(k,SPECS[k]['default'])}

    def validate(self):
        for key in ('DT_MAX_MYR','DT_MIN_MYR','ETA_FF','ETA_FB','ETA_GBS','DOMAIN_HALFSIZE_PC'):
            if self[key]<=0:raise ValueError(f'{key} must be positive')
        if self.DT_MIN_MYR>self.DT_MAX_MYR:raise ValueError('DT_MIN_MYR exceeds DT_MAX_MYR')
        for key in ('TOTAL_TIME_MYR','SOFTENING_PC','ETA_GRAD','ETA_JERK','ETA','ETA_RESTART','FLYBY_CM_EPS','TIDAL_STEP_RES'):
            if self[key]<0:raise ValueError(f'{key} cannot be negative')
        if not 2<=self.GBS_KMAX<=10:raise ValueError('GBS_KMAX must be in [2,10]')
        if self.PN_ORDER<0 or self.PN_ORDER&~63:raise ValueError('unsupported PN_ORDER bits')
        if not 0<=self.TDE_ACCRETION_FRACTION<=1:raise ValueError('TDE_ACCRETION_FRACTION must be in [0,1]')
        if not 0<=self.COLL_MASS_LOSS_C<4:raise ValueError('COLL_MASS_LOSS_C must be in [0,4)')
        if self.TIDAL_FIELD=='pointmass' and self.GALAXY_MASS_MSUN<=0:raise ValueError('pointmass tides need positive GALAXY_MASS_MSUN')
        if self.TIDAL_FIELD!='none' and self.GALAXY_R_PC<=0:raise ValueError('tidal reference radius must be positive')
