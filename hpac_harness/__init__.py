import os
import sh
import pandas as pd
import yaml
from types import SimpleNamespace
from dataclasses import dataclass
import tempfile
import sqlite3
import struct
import numpy as np
import sys
import time
import glob
import io
import re
from pathlib import Path
import os
import itertools
import yaml
from dataclasses import dataclass
from typing import Union
import itertools
import copy

APPROX_TECNHIQUE='Default'

def calc_imbalance_ratio(start, end, df):
  dd = df[start:end]
  denom = dd['RATIO'].mean()
  if denom == 0:
    return (0, 0)
  ratio = dd['APPROX'].sum() / (dd['APPROX'].sum() + dd['ACCURATE'].sum())
  return max(dd['RATIO'])/dd['RATIO'].mean(), ratio

@dataclass
class ExperimentConfiguration:
    benchmark_name: str
    items_per_thread: int
    blocksize: int
    trials: int

@dataclass
class PerfoExperimentConfiguration(ExperimentConfiguration):
    technique: str
    skip: Union[float, int]

@dataclass
class TAFExperimentConfiguration(ExperimentConfiguration):
    prediction_size: int
    history_size: int
    taf_width: int
    threshold: int

class ConfigurationGenerator:
    def __init__(self, conf=None):
        if conf:
            self.conf = conf
        else:
            self.conf = None

    def get(self):
        output = []
        vals = list(self.conf.values())

        # For itertools product, we want everything
        # to be a list, not scalars
        for v in vals:
            if isinstance(v, list):
                output.append(v)
            else:
                output.append([v])

        return output

    def combine_products(self, args):
        return itertools.product(*args,
                                 *self.get()
                                 )
    def get_labels(self):
        return list(self.conf.keys())

    def get_product(self):
        return itertools.product(*self.get())

    # combine self with other, inserting other at the end of self in order
    # return a new ConfigurationGenerator object with the intersection
    def combine_right(self, other):
        new_conf = copy.deepcopy(self.conf)
        new_conf.update(other.conf)
        return ConfigurationGenerator(conf=new_conf)

    def combine_left(self, other):
        new_conf = copy.deepcopy(other.conf)
        new_conf.update(self.conf)
        return ConfigurationGenerator(conf=new_conf)

class PerfoConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, arguments):
        # assume Python >= 3.7, where dictionaries are
        # ordered by insertion
        self.conf = {'technique': arguments['techniques'],
                     'skip': arguments['params']['skip']
                     }

class iACTConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, arguments):
        # assume Python >= 3.7, where dictionaries are
        # ordered by insertion
        self.conf = arguments

class TAFConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, arguments):
        # assume Python >= 3.7, where dictionaries are
        # ordered by insertion
        self.conf = arguments


class BenchmarkConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, name, tech_config : ConfigurationGenerator, regions: list[str],
                 overrides = None, tech_overrides = None
                 ):
        if len(regions) == 0:
            regions = ["none"]
        self.conf = {'benchmark': name,
                     'region': regions
                     }
        self.conf.update(tech_config.conf)
        if overrides:
            self.conf.update(overrides)
        if tech_overrides:
            self.conf.update(tech_overrides)


class GeneralConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, approx_type, trials, items_per_thread, blocksize):
        self.conf = {'approx_type': approx_type,'trials': trials,
                     'items_per_thread': items_per_thread,
                     'blocksize': blocksize
                     }

def calc_mape(exact, approx):
  diff = np.sum(exact - approx)
  perc = np.abs(diff / np.sum(exact))
  return perc * 100

def calc_mcr(exact_file, approx_file):
  with open(exact_file, 'r') as ef, open(approx_file, 'r') as af:
    n = 0
    unequal = 0
    for exact, approx in zip(ef, af):
      exact = int(exact.strip())
      approx = int(approx.strip())
      n += 1
      unequal += exact != approx
    return (1/n)*unequal

# The 'approx params' class will set up the environment
# for an approximation technique before running the experiment
class HPACApproxParams:
    def __init__(self, config):
        pass
    def get_name(self):
        pass
    @classmethod
    def create(cls, name, param, approx_args):
      global APPROX_TECHNIQUE

      if name in approx_args:
          approx_args = approx_args[name]
      if name == 'perfo':
        tech = param['technique']
        if tech == 'small':
            APPROX_TECHNIQUE = 'perfo:small'
            return SmallPerfoApproxParams(param)
        elif tech == 'large':
            APPROX_TECHNIQUE = 'perfo:large'
            return LargePerfoApproxParams(param)
      elif name == 'iact':
        APPROX_TECHNIQUE = 'iact'
        return iACTApproxParams(param, approx_args)
      elif name == 'taf':
        APPROX_TECHNIQUE = 'taf'
        return TAFApproxParams(param, approx_args)
      else:
        raise ValueError(f"Incorrect perfo type: {name}")

    def configure_environment(self):
        pass

    def get_technique_tuple(self):
        pass

    def get_technique_arg(self):
        pass

class HPACRuntimeEnvironment:
    def __init__(self, threads_per_block, num_blocks, num_cpu_threads):
        self.tpb = int(threads_per_block)
        self.num_blocks = int(num_blocks)
        self.num_threads = int(threads_per_block * num_blocks)
        self.num_cpu_threads = num_cpu_threads

    def configure_environment(self, items_per_thread, tpb=None, num_blocks=None):
      self.items_per_thread = int(items_per_thread)
      if tpb:
          self.tpb = tpb
      if num_blocks:
          self.num_blocks = num_blocks

      num_threads = self.tpb * self.num_blocks
      os.environ['THREADS_PER_BLOCK'] = str(self.tpb)
      os.environ['NUM_BLOCKS'] = str(self.num_blocks)
      os.environ['NUM_THREADS'] = str(num_threads)
      os.environ['OMP_PROC_BIND'] = 'true'
      os.environ['OMP_PLACES'] = 'cores'
      os.environ['OMP_NUM_THREADS'] = str(self.num_cpu_threads)

    @classmethod
    def calc_num_blocks(cls, items_per_thread, block_size, n):
        items_per_thread, block_size, n = map(int, (items_per_thread, block_size, n))

        return max(1, int((n//items_per_thread)//block_size))

    def get_db_info(self):
        return (self.items_per_thread, self.tpb,
                self.num_blocks, self.num_threads
                )


class PerfoApproxParams(HPACApproxParams):
    def __init__(self):
        self.name = "perfo"
        self.skip = -1

    # todo: specialize this method in the ini/fini/rand case
    def configure_environment(self):
        os.environ['PERFO_SKIP'] = str(self.skip)

    def get_technique_arg(self):
        return self.skip

    def get_db_info(self):
        return (self.name, self.skip)
    def get_name(self):
        return self.name

    def get_hpac_build_params(self):
      return dict()

class SmallPerfoApproxParams(PerfoApproxParams):
    def __init__(self, param):
        self.skip = int(param['skip'])
        self.name = "small"

    def get_technique_tuple(self):
        return ("sPerfo" , "perfo(small:%s)")


class LargePerfoApproxParams(PerfoApproxParams):
    def __init__(self, param):
        self.skip = int(param['skip'])
        self.name = "large"

    def get_technique_tuple(self):
        return ("lPerfo", "perfo(large:%s)")

class iACTApproxParams(HPACApproxParams):
  def __init__(self, param, approx_args):
    self.name = "iact"
    if 'hierarchy' in param:
        self.hierarchy = param['hierarchy']
    else:
        self.hierarchy = 'thread'
    self.warpsize = int(param['warp_size'])
    self.rp = param['replacement_policy']
    self.tpw = int(param['tables_per_warp'])
    self.threshold = float(param['threshold'])
    self.tsize = int(param['table_size'])
    self.ninputs = int(approx_args['input_num_items'])
    self.noutputs = int(approx_args['output_num_items'])
    self.blocksize = int(param['blocksize'])

  # todo: is this needed?
  def get_technique_arg(self):
    return self.hierarchy

  def get_technique_tuple(self):
    return ("MEMO_IN", "memo(in:%s)")

  def configure_environment(self):
    os.environ['TABLE_SIZE'] = str(self.tsize)
    os.environ[self.rp] = str(1)
    os.environ['THRESHOLD'] = str(self.threshold)
    os.environ['INPUT_ENTRY_SIZE'] = str(self.ninputs)
    os.environ['OUTPUT_ENTRY_SIZE'] = str(self.noutputs)

  def get_table_size(self):
    tables_per_block = (self.blocksize // self.warpsize) * self.tpw
    table_size_bytes= self.ninputs * 4 * self.tsize
    ts_per_block = tables_per_block * table_size_bytes
    return int(ts_per_block)

  def get_hpac_build_params(self):
    return {
      'TABLES_PER_WARP': str(self.tpw),
      'SHARED_MEMORY_SIZE': str(self.get_table_size())
      }

  def get_db_info(self):
    return (self.tsize, self.threshold, self.tpw, self.rp, self.hierarchy)
  def get_name(self):
    return self.name

class TAFApproxParams(HPACApproxParams):
  def __init__(self, param, approx_args):
    self.name = "taf"
    if 'hierarchy' in param:
        self.hierarchy = param['hierarchy']
    else:
        self.hierarchy = 'thread'
    self.warpsize = int(param['warp_size'])
    self.threshold = float(param['threshold'])
    self.hsize = int(param['history_size'])
    self.psize = int(param['prediction_size'])
    self.tw = int(param['taf_width'])
    self.ninputs = int(approx_args['input_num_items'])
    self.noutputs = int(approx_args['output_num_items'])
    self.blocksize = int(param['blocksize'])

  # todo: is this needed?
  def get_technique_arg(self):
    return self.hierarchy

  def get_technique_tuple(self):
    return ("MEMO_OUT", "memo(out:%s)")

  def configure_environment(self):
    os.environ['THRESHOLD'] = str(self.threshold)
    os.environ['HISTORY_SIZE'] = str(self.hsize)
    os.environ['PREDICTION_SIZE'] = str(self.psize)
    os.environ['INPUT_ENTRY_SIZE'] = str(self.ninputs)
    os.environ['OUTPUT_ENTRY_SIZE'] = str(self.noutputs)

  def get_hpac_build_params(self):
    return {
        'TAF_WIDTH': str(self.tw),
        'SHARED_MEMORY_SIZE': str(self.get_table_size()),
        'MAX_HIST_SIZE': str(self.hsize)
      }

  def get_table_size(self):
      # 8*NTHREADS_PER_WARP*MAX_HIST_SIZE*n_output_values
      tables_per_block = (self.blocksize // self.warpsize)
      table_size_bytes= self.noutputs * 4 * self.hsize * self.warpsize
      ts_per_block = tables_per_block * table_size_bytes
      return int(ts_per_block)

  def get_db_info(self):
    return (self.threshold, self.hsize, self.psize, self.tw, self.hierarchy)
  def get_name(self):
    return self.name

class HPACBenchmarkInstance:
    def __init__(self, name, region, config_dict, install_location = None):
      self.name = name
      self.region = region
      self.install_location = install_location
      self.error_metric = None
      self.command = None
    @classmethod
    def get_instance_class(cls, name):
      if name == 'kmeans':
        return HPACKmeansInstance
      elif name == 'blackscholes':
        return HPACBlackscholesInstance
      elif name == 'binomialoptions':
        return HPACBinomialOptionsInstance
      elif name == 'leukocyte':
        return HPACLeukocyteInstance
      elif name == 'lulesh':
          return HPACLULESHInstance
      elif name == 'lavaMD':
          return HPACLavaMDInstance
      elif name == 'lavaMD_stats':
          return HPACLavaMDStatsCollectingInstance
      elif name == 'miniFE':
        return HPACMiniFEInstance
      else:
        raise ValueError("No instance type for benchmark type "
                          f"{name} found."
                          )
    def get_name(self):
      return self.name

    def write_info_to_db(self, db_conn, exp_num):
      pass

    def get_region(self):
        return self.region

    def get_benchmark_directory(self):
      return Path(self.build_params.benchmark_directory)
    def get_n(self):
        pass
    def get_run_command(self):
        pass
    def get_runtime(self, stdout):
        pass
    def get_error(self, accurate, approx):
        pass
    def get_exact_filepath(self):
        return Path(self.run_params.exact_results)
    def get_build_location(self):
        if self.install_location:
            build_dir = self.install_location
        else:
            build_dir = "./"
        return (Path(build_dir) / self.get_benchmark_directory().stem).resolve()
    def get_db_info(self):
      region = self.get_region()
      if not region:
        region = "none"
      else:
        region = region[0].label
      return (self.get_name(), region, self.get_error_type())
    def get_error_type(self):
        return self.error_metric
    def get_executable_name(self):
      return self.build_params.executable_name
    def get_approx_filepath(self):
        return self.get_build_location() / self.build_params.output_filename
    def run_trial(self):
        with sh.pushd(self.get_build_location()):
            stdout = self.get_run_command()()
        return stdout
       
    def set_run_command(self, new_cmd):
       self.command = new_cmd



class HPACBinomialOptionsInstance(HPACBenchmarkInstance):
  @dataclass
  class RunParams:
    input_data: str
    exact_results: str
    num_steps: str
    num_options: int
    seed:int

  @dataclass
  class BuildParams:
  # TODO: this class can be extracted? Code is shared
    executable_name: str
    benchmark_directory: str
    output_filename: str

  def __init__(self, name, region, config_dict, install_location=None):
    super().__init__(name, region, config_dict, install_location)
    self.command = None
    run_config = config_dict['executable_arguments']
    self.run_params = self.RunParams(config_dict['input_data'],
                                     config_dict['exact_results'],
                                     run_config['num_steps'],
                                     run_config['num_options'],
                                     run_config['seed']
                                     )
    self.build_params = self.BuildParams(config_dict['executable_name'],
                                         config_dict['benchmark_directory'],
                                         config_dict['output_filename']
                                         )
    self.install_location = install_location
    self.runtime = 0
    self.error_metric = "mape"

  def get_run_command(self):
    if not self.command:
      exe = self.get_build_location() / Path(self.build_params.executable_name)
      exe = exe.resolve()
      runp = self.run_params
      self.command =  sh.Command(exe).bake(self.run_params.input_data,
                                           self.run_params.num_steps,
                                           self.run_params.seed,
                                           self.build_params.output_filename
                                           )
    return self.command

  def build(self, pre=None, post=None):
    # TODO: this can probably be moved to the baseclass
    build_dir = self.get_build_location()
    if not build_dir.exists():
        build_dir.mkdir(parents=True, exist_ok=True)
    sh.cp('-r',
          list(self.get_benchmark_directory().glob('*')),
          build_dir
          )

    if pre:
        pre()
    builder = UNIXMakeProgramBuilder(build_dir, 'Makefile.approx')
    builder.build()
    if post:
        post()

  def get_runtime(self, stdout):
    stdout_lines = stdout.split('\n')
    runtime = filter(lambda x: x.startswith('binomialOptionsOffload()'),
                     stdout_lines
                     )
    runtime = list(runtime)

    return float(runtime[0].split()[2])

  def get_error(self):
    exact_file = self.get_exact_filepath()
    approx_file = self.get_approx_filepath()
    with open(exact_file, 'rb') as exact, open(approx_file, 'rb') as approx:
      exact_data = exact.read()
      approx_data = approx.read()

      e_len, e_type = struct.unpack('@QI', exact_data[0:12])
      a_len, a_type = struct.unpack('@QI', approx_data[0:12])

      assert (e_len == a_len) and (e_type == a_type), "Exact and approx files do not have same type or length"

      e_data = np.frombuffer(exact_data[12::], dtype=np.float64)
      a_data = np.frombuffer(approx_data[12::], dtype=np.float64)
      mape = calc_mape(e_data, a_data)
      return mape

  def get_n(self):
    return self.run_params.num_options

class HPACBlackscholesInstance(HPACBenchmarkInstance):
  @dataclass
  class RunParams:
    input_data: str
    exact_results: str
    num_threads: str
    num_options: int

  @dataclass
  class BuildParams:
  # TODO: this class can be extracted? Code is shared
    executable_name: str
    benchmark_directory: str
    output_filename: str

  def __init__(self, name, region, config_dict, install_location=None):
    super().__init__(name, region, config_dict, install_location)
    self.command = None
    run_config = config_dict['executable_arguments']
    self.run_params = self.RunParams(config_dict['input_data'],
                                     config_dict['exact_results'],
                                     '1',
                                     run_config['num_options']
                                     )
    self.build_params = self.BuildParams(config_dict['executable_name'],
                                         config_dict['benchmark_directory'],
                                         config_dict['output_filename']
                                         )
    self.install_location = install_location
    self.runtime = 0
    self.error_metric = "mape"

  def get_run_command(self):
    if not self.command:
      exe = self.get_build_location() / Path(self.build_params.executable_name)
      exe = exe.resolve()
      runp = self.run_params
      self.command =  sh.Command(exe).bake( self.run_params.num_threads,
                                            self.run_params.input_data,
                                            self.build_params.output_filename
                                           )
    return self.command

  def build(self, pre=None, post=None):
      build_dir = self.get_build_location()
      if not build_dir.exists():
          build_dir.mkdir(parents=True, exist_ok=True)
      sh.cp('-r', glob.glob(f'{self.build_params.benchmark_directory}/*'), build_dir)
      if pre:
          pre()
      builder = UNIXMakeProgramBuilder(build_dir, 'Makefile.approx')
      builder.build()
      if post:
          post()

  def get_runtime(self, stdout):
    stdout_lines = stdout.split('\n')
    runtime = list(filter(lambda x: x.startswith('Elapsed'), stdout_lines))
    return float(runtime[0].split()[1])

  def get_error(self):
    exact_file = self.get_exact_filepath()
    approx_file = self.get_approx_filepath()
    with open(exact_file, 'rb') as exact, open(approx_file, 'rb') as approx:
      exact_data = exact.read()
      approx_data = approx.read()

      e_len, e_type = struct.unpack('@QI', exact_data[0:12])
      a_len, a_type = struct.unpack('@QI', approx_data[0:12])

      assert (e_len == a_len) and (e_type == a_type), "Exact and approx files do not have same type or length"

      e_data = np.frombuffer(exact_data[12::], dtype=np.float64)
      a_data = np.frombuffer(approx_data[12::], dtype=np.float64)
      mape = calc_mape(e_data, a_data)
      return mape

  def get_n(self):
    return self.run_params.num_options


class HPACLeukocyteInstance(HPACBenchmarkInstance):
    @dataclass
    class RunParams:
        input_data: str
        exact_results: str
        num_cells: int
        num_frames: int

    @dataclass
    class BuildParams:
        executable_name: str
        benchmark_directory: str
        output_filename: str

    def __init__(self, name, region, config_dict, install_location=None):
        super().__init__(name, region, config_dict, install_location)
        self.command = None
        run_config = config_dict['executable_arguments']
        self.run_params = self.RunParams(config_dict['input_data'],
                                         config_dict['exact_results'],
                                         run_config['num_cells'],
                                         run_config['num_frames']
                                    )
        self.build_params = self.BuildParams(config_dict['executable_name'],
                                             config_dict['benchmark_directory'],
                                             config_dict['output_filename']
                                             )
        self.install_location = install_location

        self.runtime = 0
        self.error_metric = "mape"

    def get_run_command(self):
        if not self.command:
            exe = self.get_build_location() / Path(self.build_params.executable_name)
            exe = exe.resolve()
            runp = self.run_params
            self.command =  sh.Command(exe).bake(self.run_params.input_data,
                                                 self.run_params.num_frames
                                                 )
        return self.command

    # TODO: this can probably be moved to the baseclass
    def build(self, pre=None, post=None):
        build_dir = self.get_build_location()
        if not build_dir.exists():
            build_dir.mkdir(parents=True, exist_ok=True)
        sh.cp('-r', glob.glob(f'{self.build_params.benchmark_directory}/*'), build_dir)
        if pre:
            pre()
        builder = UNIXMakeProgramBuilder(build_dir, 'Makefile.approx')
        builder.build()
        if post:
            post()

    # Given the stdout for this benchmark, return the runtime
    def get_runtime(self, stdout):
        stdout_lines = stdout.split('\n')
        runtime = list(filter(lambda x: x.startswith('Total application run time:'),
                              stdout_lines)
                       )
        return float(runtime[0].split()[4])

    # given accurate and approx outputs, return error metric
    def get_error(self):
        e_file = open(self.get_exact_filepath(), 'r')
        a_file = open(self.get_approx_filepath(), 'r')

        exact = []
        approx = []
        for line in e_file:
          spl = line.strip().split(',')
          d1 = float(spl[1])
          d2 = float(spl[2])
          exact.append(d1)
          exact.append(d2)

        for line in a_file:
          spl = line.strip().split(',')
          d1 = float(spl[1])
          d2 = float(spl[2])
          approx.append(d1)
          approx.append(d2)


        exact = np.array(exact, dtype=np.float64)
        approx = np.array(approx, dtype=np.float64)

        return calc_mape(exact, approx)

    # TODO: this should be used when determining number of blocks
    def get_n(self):
        return self.run_params.cells

class HPACLULESHInstance(HPACBenchmarkInstance):
    @dataclass
    class RunParams:
        size: int
        max_iterations: int
        exact_results: str


    @dataclass
    class BuildParams:
        executable_name: str
        benchmark_directory: str

    def __init__(self, name, region, config_dict, install_location=None):
        super().__init__(name, region, config_dict, install_location)
        self.command = None
        run_config = config_dict['executable_arguments']
        self.run_params = self.RunParams(
                                         run_config['size'],
                                         run_config['max_iterations'],
                                         config_dict['exact_results']
                                    )
        self.build_params = self.BuildParams(config_dict['executable_name'],
                                             config_dict['benchmark_directory']
                                             )
        self.install_location = install_location

        self.runtime = 0
        self.error_metric = "mape"

    def get_run_command(self):
        if not self.command:
            exe = self.get_build_location() / Path(self.build_params.executable_name)
            exe = exe.resolve()
            runp = self.run_params
            self.command =  sh.Command(exe).bake('-s', self.run_params.size,
                                                 '-i', self.run_params.max_iterations
                                                 )
        return self.command

    # TODO: this can probably be moved to the baseclass
    def build(self, pre=None, post=None):
        build_dir = self.get_build_location()
        if not build_dir.exists():
            build_dir.mkdir(parents=True, exist_ok=True)
        sh.cp('-r', glob.glob(f'{self.build_params.benchmark_directory}/*'), build_dir)
        if pre:
            pre()
        builder = UNIXMakeProgramBuilder(build_dir, 'Makefile.approx')
        builder.build()
        if post:
            post()

    # Given the stdout for this benchmark, return the runtime
    def get_runtime(self, stdout):
        stdout_lines = stdout.split('\n')
        runtime = list(filter(lambda x: x.startswith('Elapsed time'),
                              stdout_lines)
                       )
        # HACK: the error in this benchmark comes from stdout, not file
        self._stdout = stdout
        return float(runtime[0].split()[3])

    def _get_energy(self, stdout):
        stdout_lines = stdout.split('\n')
        runtime = list(filter(lambda x: x.strip().startswith('Final Origin Energy'),
                              stdout_lines)
                       )
        return float(runtime[0].split()[4])

    # given accurate and approx outputs, return error metric
    def get_error(self):
        assert self._stdout, "For this benchmark, get_runtime MUST be called before get_error"
        e_file = open(self.get_exact_filepath(), 'r')

        a_energy = self._get_energy(self._stdout)
        approx = [a_energy]
        exact = [float(e_file.read().split('\n')[0])]
        exact = np.array(exact, dtype=np.float64)
        approx = np.array(approx, dtype=np.float64)

        return calc_mape(exact, approx)

    # TODO: this should be used when determining number of blocks
    def get_n(self):
        return self.run_params.size**3



class HPACLavaMDInstance(HPACBenchmarkInstance):
    @dataclass
    class RunParams:
        boxes1d: int
        exact_results: str


    @dataclass
    class BuildParams:
        executable_name: str
        benchmark_directory: str
        output_filename: str

    def __init__(self, name, region, config_dict, install_location=None):
        super().__init__(name, region, config_dict, install_location)
        self.command = None
        run_config = config_dict['executable_arguments']
        self.run_params = self.RunParams(
                                         run_config['boxes1d'],
                                         config_dict['exact_results']
                                    )
        self.build_params = self.BuildParams(config_dict['executable_name'],
                                             config_dict['benchmark_directory'],
                                             config_dict['output_filename']
                                             )
        self.install_location = install_location

        self.runtime = 0
        self.error_metric = "mape"

    def get_run_command(self):
        if not self.command:
            exe = self.get_build_location() / Path(self.build_params.executable_name)
            exe = exe.resolve()
            runp = self.run_params
            self.command =  sh.Command(exe).bake('-boxes1d', self.run_params.boxes1d)
        return self.command

    # TODO: this can probably be moved to the baseclass
    def build(self, pre=None, post=None):
        build_dir = self.get_build_location()
        if not build_dir.exists():
            build_dir.mkdir(parents=True, exist_ok=True)
        sh.cp('-r', glob.glob(f'{self.build_params.benchmark_directory}/*'), build_dir)
        if pre:
            pre()
        builder = UNIXMakeProgramBuilder(build_dir, 'Makefile.approx')
        builder.build()
        if post:
            post()

    # Given the stdout for this benchmark, return the runtime
    def get_runtime(self, stdout):
        stdout_lines = stdout.split('\n')
        runtime = list(filter(lambda x: x.startswith('Device offloading time'),
                              stdout_lines)
                       )
        return float(runtime[0].split()[3])

    # given accurate and approx outputs, return error metric
    def get_error(self):
        exact_file = self.get_exact_filepath()
        approx_file = self.get_approx_filepath()
        with open(exact_file, 'rb') as exact, open(approx_file, 'rb') as approx:
            exact_data = exact.read()
            approx_data = approx.read()

            e_len = struct.unpack('@L', exact_data[0:8])
            a_len = struct.unpack('@L', approx_data[0:8])

            assert (e_len == a_len), "Exact and approx files do not have same length"

            e_data = np.frombuffer(exact_data[8::], dtype=np.float64)
            a_data = np.frombuffer(approx_data[8::], dtype=np.float64)
            mape = calc_mape(e_data, a_data)
        return mape
    # TODO: this should be used when determining number of blocks
    def get_n(self):
        return self.run_params.size**3

class HPACLavaMDStatsCollectingInstance(HPACLavaMDInstance):
    def __init__(self, name, region, config_dict, install_location=None):
        super().__init__(name, region, config_dict, install_location)
    def get_imbalance(self):
        # TODO: Base StatsCollectingClass that has get_stats_filepath like we have
        file_location = self.get_approx_filepath().parent
        file_name = Path('thread_stats.csv')
        approx_info = pd.read_csv(file_location / file_name)
        # NOTE: Bad to assume fixed warp size. Oh well
        imb_all = np.ndarray((min(10000,len(approx_info)//32), 3), dtype=np.float64)
        for warp_start in range(0,min(10000*32,len(approx_info)), 32):
            warp_num = warp_start // 32
            imb, ratio = calc_imbalance_ratio(warp_start, warp_start+32, approx_info)
            imb_all[warp_num] = [int(warp_num), float(imb), float(ratio)]
        return imb_all

class HPACKmeansInstance(HPACBenchmarkInstance):
    @dataclass
    class RunParams:
        input_data: str
        exact_results: str
        num_points: int
        num_features: int
        min_clusters: int
        max_clusters: int
        num_iters: int

    @dataclass
    class BuildParams:
        executable_name: str
        benchmark_directory: str
        output_filename: str

    def __init__(self, name, region, config_dict, install_location=None):
        super().__init__(name, region, config_dict, install_location)
        self.command = None
        self.nconverged = 0
        run_config = config_dict['executable_arguments']
        self.run_params = self.RunParams(config_dict['input_data'],
                                         config_dict['exact_results'],
                                         run_config['num_points'],
                                         run_config['num_features'],
                                         run_config['min_clusters'],
                                         run_config['max_clusters'],
                                         run_config['num_kmeans_iters']
                                    )
        self.build_params = self.BuildParams(config_dict['executable_name'],
                                             config_dict['benchmark_directory'],
                                             'assignments_approx.txt'
                                             )
        self.install_location = install_location

        self.runtime = 0
        self.error_metric = "mcr"

    def get_run_command(self):
        if not self.command:
            exe = self.get_build_location() / Path(self.build_params.executable_name)
            exe = exe.resolve()
            runp = self.run_params
            self.command =  sh.Command(exe).bake('-i', runp.input_data,
                                                 '-m', runp.min_clusters,
                                                 '-n', runp.max_clusters,
                                                 '-l', runp.num_iters
                                                 )
        return self.command

    def build(self, pre=None, post=None):
        build_dir = self.get_build_location()
        if not build_dir.exists():
            build_dir.mkdir(parents=True, exist_ok=True)
        sh.cp('-r', glob.glob(f'{self.build_params.benchmark_directory}/*'), build_dir)
        if pre:
            pre()
        builder = UNIXMakeProgramBuilder(build_dir, 'Makefile.approx')
        builder.build()
        if post:
            post()

    # Given the stdout for this benchmark, return the runtime
    def get_runtime(self, stdout):
        stdout_lines = stdout.split('\n')
        nloops = list(filter(lambda x: x.startswith('Kmeans converged'), stdout_lines))
        self.nloops = int(nloops[0].split()[4])
        runtime = list(filter(lambda x: x.startswith('Kmeans core'), stdout_lines))
        return float(runtime[0].split()[3])

    def write_info_to_db(self, db_conn, exp_num):
      global APPROX_TECHNIQUE
      table_name = f'kmeans_{APPROX_TECHNIQUE}'
      cur = db_conn.cursor()
      cur.execute(f'create table if not exists {table_name} (exp_num integer, num_converged integer)')
      cur.execute(f'INSERT INTO {table_name} VALUES(?, ?)', (exp_num, self.nloops))
      db_conn.commit()

    # given accurate and approx outputs, return error metric
    def get_error(self):
        return calc_mcr(self.get_exact_filepath(), self.get_approx_filepath())

    # TODO: this should be used when determining number of blocks
    def get_n(self):
        return self.run_params.num_points


class HPACMiniFEInstance(HPACBenchmarkInstance):
    @dataclass
    class RunParams:
        exact_results: str
        num_rows: int
        nx: int
        ny: int
        nz: int

    @dataclass
    class BuildParams:
        executable_name: str
        benchmark_directory: str

    def __init__(self, name, region, config_dict, install_location=None):
        super().__init__(name, region, config_dict, install_location)
        self.command = None
        run_config = config_dict['executable_arguments']
        self.run_params = self.RunParams(config_dict['exact_results'],
                                         run_config['num_rows'],
                                         run_config['nx'],
                                         run_config['ny'],
                                         run_config['nz']
                                    )
        self.build_params = self.BuildParams(config_dict['executable_name'],
                                             config_dict['benchmark_directory']
                                             )
        self.install_location = install_location

        self.runtime = 0
        self.error_metric = "mape"

    def get_run_command(self):
        if not self.command:
            exe = self.get_build_location() / Path(self.build_params.executable_name)
            exe = exe.resolve()
            runp = self.run_params
            self.command =  sh.Command(exe).bake('-nx', runp.nx,
                                                 '-ny', runp.ny,
                                                 '-nz', runp.nz
                                                 )
        return self.command

    def build(self, pre=None, post=None):
        build_dir = self.get_build_location()
        if not build_dir.exists():
            build_dir.mkdir(parents=True, exist_ok=True)
        sh.cp('-r', glob.glob(f'{self.build_params.benchmark_directory}/*'), build_dir)
        if pre:
            pre()
        builder = UNIXMakeProgramBuilder(build_dir, 'Makefile.approx')
        builder.build()
        if post:
            post()

    # Given the stdout for this benchmark, return the runtime
    def get_runtime(self, stdout):
        stdout_lines = stdout.split('\n')
        runtime = list(filter(lambda x: x.startswith('SOLVE TIME'), stdout_lines))
        self._stdout = stdout_lines
        return float(runtime[0].split()[2])

    def write_info_to_db(self, db_conn, exp_num):
      pass

    # given accurate and approx outputs, return error metric
    def get_error(self):
      error = list(filter(lambda x: x.startswith('Final Resid Norm'), self._stdout))
      error = error[0].split()[3]
      approx_residual = np.array(float(error))
      exact_file = open(self.get_exact_filepath(), 'r')
      exact_residual = float(exact_file.read().strip())
      exact_file.close()
      return calc_mape(exact_residual, approx_residual)

    # TODO: this should be used when determining number of blocks
    def get_n(self):
        return self.run_params.num_rows

class ProgramBuilder:
    def __init__(self, build_directory):
        # check that the path exists
        self.build_dir = Path(build_directory)
        self.build_command = None
        self.target = None
    def build(self):
        # TODO: Errors
      try:
        self.build_command()
      except Exception as e:
        print(e.stdout)
        print(e.stderr)
        sys.exit(2)
    def configure(self):
        pass
    def get_name(self):
        return "None"
    def set_target(self, new_target):
        self.target = new_target


class CMakeBuildConfig:
    def __init__(self, options):
        self.opts = options
        self.clang_version = options['PACKAGE_VERSION']
    def get_opts_for_build(self):
      opts = list()
      for o,v in self.opts.items():
        opts.append(f'-D{o}={str(v)}')
      return opts
    def add_opt(self, key, value):
      self.opts[key] = value



class CMakeProgramBuilder(ProgramBuilder):
    def __init__(self, build_directory, destination_directory, config, cmake_generator):
        super().__init__(build_directory)
        self.config = config
        self.destination = Path(destination_directory)
        self.generator = cmake_generator
        self._configured = False
    def configure(self):
        self.destination.mkdir(parents=True, exist_ok=True)
        with sh.pushd(self.destination):
            cmd = sh.cmake.bake('-G', self.generator.get_name(),
                                *self.config.get_opts_for_build(),
                                f'{self.build_dir}'
                                )
            cmd()
        self._configured = True

    def install(self):
        assert self._configured
        self.generator.set_target('install')
        self.generator.build()

class NinjaProgramBuilder(ProgramBuilder):
    def __init__(self, build_directory, target = ""):
        super().__init__(build_directory)
        self.target = target
        self.build_command = None
    def build(self):
        with sh.pushd(self.build_dir):
            if not self.build_command:
                self.build_command = self._build_command()
            return super().build()
    def _build_command(self):
        assert self.target
        return sh.ninja.bake(self.target)
    def get_name(self):
        return "Ninja"

class UNIXMakeProgramBuilder(ProgramBuilder):
    def __init__(self, build_directory, makefile_name, target = ""):
        super().__init__(build_directory)
        self.makefile_name = makefile_name
        self.target = target
        self.build_command = None
    def build(self):
        with sh.pushd(self.build_dir):
            if not self.build_command:
                self.build_command = self._build_command(self.build_command)
            return super().build()
    def _build_command(self, cmd):
        sh.make('clean')
        if self.target:
            return sh.make.bake('-f', self.makefile_name, self.target)
        else:
            return sh.make.bake('-f', self.makefile_name)
    def get_name(self):
        return "UNIX Makefiles"


class HPACNodeExperiment:
    def __init__(self, exp_num, instance, rtenv, approx_params, db_writer, num_trials):
        self.hostname = sh.hostname().strip()
        self.exp_num = exp_num
        self.timing = [0, 0]
        self.instance = instance
        self.db_writer = db_writer
        self.num_trials = num_trials
        self.trials = list()
        self.rtenv = rtenv
        self.approx_params = approx_params

    def start(self):
        self.timing[0] = time.time()

    def stop(self):
        self.timing[1] = time.time()

    def get_elapsed(self):
        return self.timing[1] - self.timing[0]

    def run_trials(self, num_trials):
        self.approx_params.configure_environment()
        cmd_str = str(self.instance.get_run_command())
        trials = list()
        self.start()
        for t in range(num_trials):
            trial_output = self.instance.run_trial()
            runtime = self.instance.get_runtime(trial_output)
            error = self.instance.get_error()
            trials.append((runtime, error))
        self.trials = trials
        self.stop()
        return trials

    def write_info_to_db(self, db_conn, table_name, trial_info = None):
        if not trial_info:
            trial_info = self.trials
        env_info = self.rtenv.get_db_info()
        inst_info = self.instance.get_db_info()
        params_info = self.approx_params.get_db_info()
        info = list()

        for tn, tinf in enumerate(trial_info):
            info.append([self.exp_num, tn+1, *env_info, *inst_info, *params_info,
                         *tinf, self.timing[0], self.timing[1], self.hostname
                         ]
                        )
        print(info)
        cur = db_conn.cursor()
        insert = ['?'] * len(info[0])
        insert = ','.join(insert)
        cur.executemany(f'INSERT INTO {table_name} VALUES({insert})', info)
        db_conn.commit()

        # Some benchmarks have extra information to write to the db.
        # For instance, in K-Means we are interested in the number of
        # iterations to convergence.
        self.instance.write_info_to_db(db_conn, self.exp_num)

class HPACMemoryUsageTrackingNodeExperiment(HPACNodeExperiment):
   # metrics is a list of string ncu metric names to track
    def __init__(self, exp_num, instance, rtenv, approx_params, db_writer, num_trials, metrics):
      super().__init__(exp_num, instance, rtenv, approx_params, db_writer, num_trials)
      self._metrics = metrics

    def run_trials(self, num_trials):
        self.approx_params.configure_environment()
        original_run_cmd = self.instance.get_run_command()
        ncu = sh.Command('ncu').bake('--metrics', ','.join(self._metrics), '-s', 1,
                      '--csv'
        )

        
        self.instance.set_run_command(ncu.bake(original_run_cmd._path).bake(*original_run_cmd._partial_baked_args))
        #self.instance.set_run_command(ncu.bake(original_run_cmd._path).bake(*original_run_cmd._partial_baked_args))
        print(self.instance.get_run_command())
        output = self.instance.run_trial()

        output_lines = output.split('\n')
        target = '"ID","Process ID","Process Name"'
        for idx, l in enumerate(output_lines):
          if l.startswith(target):
            interesting_data = output_lines[idx::]
        data_str = io.StringIO('\n'.join(interesting_data))
        df = pd.read_csv(data_str)
        df.loc[:,'exp_num'] = self.exp_num
        self.df = df[["Kernel Name", "Metric Name", "Metric Unit", "Metric Value"]]
        print(df)

    def write_info_to_db(self, db_conn, exp_num):
        env_info = self.rtenv.get_db_info()
        inst_info = self.instance.get_db_info()
        params_info = self.approx_params.get_db_info()
        info = list()

        # loop over the rows of the dataframe
        for idx, row in self.df.iterrows():
           info.append(self.exp_num, idx+1, *env_info, *inst_info, *params_info,
                       row['Kernel Name'], row['Metric Name'], row['Metric Unit'], row['Metric Value']
                       )

        print(info)
        cur = db_conn.cursor()
        insert = ['?'] * len(info[0])
        insert = ','.join(insert)
        cur.executemany(f'INSERT INTO {table_name} VALUES({insert})', info)
        db_conn.commit()
        self.instance.write_info_to_db(db_conn, self.exp_num)

class HPACStatsCollectingNodeExperiment(HPACNodeExperiment):
    def __init__(self, exp_num, instance, rtenv, approx_params, db_writer, num_trials):
        super().__init__(exp_num, instance, rtenv, approx_params, db_writer, num_trials)

    def run_trials(self, num_trials):
        self.approx_params.configure_environment()
        self.aib_info = list()
        cmd_str = str(self.instance.get_run_command())
        trials = list()
        self.start()
        for t in range(num_trials):
            trial_output = self.instance.run_trial()
            runtime = self.instance.get_runtime(trial_output)
            error = self.instance.get_error()
            aib = self.instance.get_imbalance()
            self.aib_info.append(aib)
            trials.append((runtime, error))
        self.trials = trials
        self.stop()
        return trials

    def write_info_to_db(self, db_conn, table_name, trial_info = None):
        super().write_info_to_db(db_conn, table_name, trial_info)
        info = list()
        for tn, aib in enumerate(self.aib_info):
            for warp_info in aib:
                warp_num, imbalance, ratio = warp_info
                info.append([self.exp_num, tn+1, int(warp_num), imbalance, ratio])

        cur = db_conn.cursor()
        insert = ['?'] * len(info[0])
        insert = ','.join(insert)
        cur.executemany(f'INSERT INTO warp_approx_stats VALUES({insert})', info)
        db_conn.commit()


class HPACInstaller:
    def __init__(self, hpac_location, install_location, clang_src, clang_version, enable_shared=0, device_stats = 0, sm_size=0, tables_per_warp= 0, taf_width=32, other_options = None):
        hpac_location = Path(hpac_location).resolve()
        install_location = Path(install_location).resolve()
        self.hpac_location = hpac_location
        self.install_location = install_location
        self.build_cfg = self.get_build_cfg(install_location, clang_src, clang_version,
                                            enable_shared, device_stats, sm_size,
                                            tables_per_warp, taf_width, other_options
                                            )
    def install(self, pre=None, post=None):
        if pre:
            pre()
        self.setup_build_environment()
        hpac_builder = NinjaProgramBuilder(self.install_location, 'install')
        hpac_installer = CMakeProgramBuilder(self.hpac_location / Path('approx'), self.install_location, self.build_cfg, hpac_builder)

        hpac_installer.configure()
        hpac_installer.install()
        self.setup_experiment_environment()

        if post:
            post()

    def setup_build_environment(self):
        path_re = re.compile('export PATH=(\s+|\S+):\$PATH')
        hpac_arch_re = re.compile('export HPAC_GPU_ARCH=(\S+)')
        hpac_sm_re = re.compile('export HPAC_GPU_SM=(\S+)')
        os.environ['CC'] = 'clang'
        os.environ['CPP'] = 'clang++'
        os.environ['CXX'] = 'clang++'
        opt = open(Path(self.hpac_location, 'hpac_env.sh'), 'r').readlines()
        for line in opt:
            result=path_re.search(line)
            if result != None:
                hpac_build_dir = Path(result.group(1)).parent.stem
                hpac_build_dir = Path(self.hpac_location) / hpac_build_dir
                add_to_env('PATH', str(hpac_build_dir / Path('bin')))
                add_to_env('LD_LIBRARY_PATH', str(hpac_build_dir / Path('lib')))
                continue

            result = hpac_arch_re.search(line)
            if result != None:
              hpac_arch = result.group(1)
              set_env('HPAC_GPU_ARCH', hpac_arch)
              self.build_cfg.add_opt('GPU_ARCH', hpac_arch)
              continue

            result = hpac_sm_re.search(line)
            if result != None:
              hpac_sm = result.group(1)
              set_env('HPAC_GPU_SM', hpac_sm)
              self.build_cfg.add_opt('GPU_SM', hpac_sm)
              continue


    def setup_experiment_environment(self):
        add_to_env('LD_LIBRARY_PATH', f'{self.install_location}/lib')
        add_to_env('CPATH', f'{self.install_location}/lib/clang/{self.build_cfg.clang_version}/include')
        set_env('LIBAPPROX_LOCATION', f"{self.install_location}/lib/")

    def get_build_cfg(self, destination, clang_src, clang_version, enable_shared, device_stats, sm_size, tables_per_warp, taf_width, other_options):
        options = {
            'CMAKE_INSTALL_PREFIX': destination,
            'LLVM_EXTERNAL_CLANG_SOURCE_DIR': clang_src,
            'PACKAGE_VERSION': clang_version,
            'LIBAPPROX_ENABLE_SHARED': enable_shared,
            'DEV_STATS': device_stats,
            'SHARED_MEMORY_SIZE': sm_size,
            'TABLES_PER_WARP': tables_per_warp,
            'TAF_WIDTH': taf_width
            }
        if other_options:
            options.update(other_options)
        return CMakeBuildConfig(options)

def add_to_env(variable, value):
    if variable in os.environ:
        os.environ[variable] = f"{value}:{os.environ[variable]}"
    else:
        os.environ[variable] = f"{value}"

def set_env(variable, value):
    os.environ[variable] = f"{value}"

@dataclass
class ApproxRegion:
    file: Path
    line_num: int
    inputs: str
    outputs: str
    inouts: str
    label: str
    technique: list[str]


def find_approx_regions(src_dir, src_files):
    regions = []
    for f in src_files:
        with open(f, "r", errors='ignore') as fd:
            for num, line  in enumerate(fd,1):
                if ("//@APPROX") in line:
                    inputs=""
                    outputs=""
                    inouts=""
                    label=""
                    approx_techs = []
                    if " INOUT" in line:
                        result=inout_regex.search(line)
                        inouts = "inout(" + result.group(1) + ")"
                    if " IN" in line:
                        result=in_regex.search(line)
                        if result != None:
                            inputs = "in(" + result.group(1) + ")"
                    if " OUT" in line:
                        result=out_regex.search(line)
                        outputs = "out(" + result.group(1) + ")"
                    if " LABEL" in line:
                        result = label_regex.search(line)
                        label = result.group(1)
                    if "APPROX_TECH" in line:
                        result = tech_regex.search(line)
                        if result != None:
                            approx_techs = result.group(1).split("|")
                            approx_techs = [v.replace(" ", "") for v in approx_techs]
                    regions.append( ApproxRegion(Path(f), num, inputs, outputs, inouts, label, approx_techs) )
    return regions

def apply_approx_technique(src_dir, src_files, regions, technique, param):
    approx = technique
    elem = None
    # select the technique specified by the user
    for r in regions:
      if technique[0] in r.technique:
        elem = r
        break
    assert elem, f"Specified technique '{technique[0]}' not found"
    for src in src_files:
        output_file = io.StringIO()
        with open(src, "r", errors="ignore") as fd:
            for num, line  in enumerate(fd,1):
                technique_param = None
                if (src == elem.file and num == elem.line_num):
                    if approx[0] in ["sPerfo", "lPerfo", 'MEMO_IN', 'MEMO_OUT']:
                        # output_file.write("int __approx_step__ = approx_rt_get_step();\n")
                        technique_param = f'{param}'
                    elif approx[0] in ["rPerfo", "fPerfo", "iPerfo"]:
                        raise ValueError("rPerfo, fPerfo, iPerfo not currently supported")
                        output_file.write("float __approx_percentage__ = approx_rt_get_percentage();\n")
                        technique_param = '__approx_percentage__'
                    output_file.write("#pragma approx ")
                    if technique_param:
                      output_file.write(approx[1] % technique_param)
                    else:
                      output_file.write(approx[1])
                    if approx[0] not in ["sPerfo", "lPerfo", "rPerfo", "iPerfo", "fPerfo"]:
                        output_file.write(elem.inputs + " " + elem.outputs + " " + elem.inouts + " ")
                        output_file.write(" label(\"" +  elem.label + "\")\n")
                    else:
                        output_file.write("\n")
                else:
                    output_file.write(line)
        output_file.seek(0)
        with open(src, 'w') as fd:
            fd.write(output_file.read())

approximate_techniques = [("iACT", "memo(in:%s)"), ("TAF", "memo(out:%s)"), ("sPerfo" , "perfo(small:%s)"), ("lPerfo", "perfo(large:%s)"), ("rPerfo", "perfo(rand:__approx_percentage__)"), ("iPerfo",  "perfo(init:%s)"), ("fPerfo", "perfo(fini:%s)") ]

IN_PATTERN='IN\((.*?)\)\s+'
OUT_PATTERN='OUT\((.*?)\)\s+'
INOUT_PATTERN='INOUT\((.*?)\)\s+'
LABEL_PATTERN='LABEL\(\"(.*?)\"\)\s+'
APPROX_TECH_PATTERN='APPROX_TECH\((.*?)\)\s+'

in_regex=re.compile(IN_PATTERN)
out_regex=re.compile(OUT_PATTERN)
inout_regex=re.compile(INOUT_PATTERN)
label_regex=re.compile(LABEL_PATTERN)
tech_regex = re.compile(APPROX_TECH_PATTERN)

def applyTechOnRegion(tech, r_tech):
    if tech == "iACT" and "MEMO_IN" in r_tech:
        return True
    elif tech == "TAF" and "MEMO_OUT" in r_tech:
        return True
    elif tech == "sPerfo" and "PERFO" in r_tech:
        return True;
    elif tech == "lPerfo" and "PERFO" in r_tech:
        return True;
    elif tech == "rPerfo" and "PERFO" in r_tech:
        return True;
    return False
