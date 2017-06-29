-- To Do List

-----------------------------------------------------------
-- Table Schema
set search_path to asamba, public;

-- Changing the default attributes of this database
set shared_buffers='2GB';
set work_mem='10MB';
set synchronous_commit='off';
set full_page_writes='off';

-----------------------------------------------------------
-- Function Definitions and Operator Overloadings
create or replace function approximately_equals_to (
  in leftarg real, in rightarg real) returns boolean
  as $$ select abs(leftarg - rightarg) <= 1e-5 * abs(leftarg) $$
  language sql stable strict;

create operator ~ ( 
  procedure = approximately_equals_to,
  leftarg   = real,
  rightarg  = real
);

-----------------------------------------------------------

-- Create Tables
-----------------------------------------------------------
create table tracks (
  id             serial,
  M_ini          real not null,
  fov            real not null,
  Z              real not null,
  logD           real not null,

  --

  primary key (id),

  unique (M_ini, fov, Z, logD),

  constraint positive_mass check (M_ini > 0),
  constraint positive_ov check (fov >= 0),
  constraint positive_Z check (Z > 0),
  constraint positive_log_D check (logD >= 0)

);

-- -- create index index_track_id on tracks (id asc);  
-- create index index_M_ini on tracks (M_ini asc);
-- create index index_fov on tracks (fov asc);
-- create index index_Z on tracks (Z asc);
-- create index index_log_D on tracks (logD asc);

-----------------------------------------------------------
create table models (
  id             serial,
  id_track       int not null,
  Xc             real not null,
  model_number   int not null,

  star_mass      real,
  radius         real,
  log_Teff       real,
  log_g          real,
  log_L          real,
  log_Ledd       real,
  log_abs_mdot   real,
  mass_conv_core real,

  star_age       real,
  dynamic_timescale real,
  kh_timescale   real,
  nuc_timescale  real,

  log_center_T   real,
  log_center_Rho real,
  log_center_P   real,

  center_h1      real,
  center_h2      real,
  center_he3     real,
  center_he4     real,
  center_c12     real,
  center_c13     real,
  center_n14     real,
  center_n15     real,
  center_o16     real,
  center_o18     real,
  center_ne20    real,
  center_ne22    real,
  center_mg24    real,

  surface_h1     real,
  surface_h2     real,
  surface_he3    real,
  surface_he4    real,
  surface_c12    real,
  surface_c13    real,
  surface_n14    real,
  surface_n15    real,
  surface_o16    real,
  surface_o18    real,
  surface_ne20   real,
  surface_ne22   real,
  surface_mg24   real,

  delta_nu       real,
  nu_max         real,
  acoustic_cutoff real,
  delta_Pg       real,

  Mbol           real,
  bcv            real,
  U_B            real,
  B_V            real,
  V_R            real,
  V_I            real,
  V_K            real,
  R_I            real,
  I_K            real,
  J_H            real,
  H_K            real,
  K_L            real,
  J_K            real,
  J_L            real,
  J_Lp           real,
  K_M            real,

  --

  primary key (id),
  -- foreign key (id_track) references tracks (id),

  constraint positive_Xc check (Xc >= 0),
  check (model_number >= 0)

);

-- -- create index index_models_id on models (id asc);
-- create index index_models_id_track on models (id_track);
-- create index index_logTeff_logg on models (log_Teff, log_g);
-- create index index_Xc on models (Xc desc);
-- create index index_age on models (star_age asc);
-- create index index_delta_Pg on models (delta_Pg desc);

-----------------------------------------------------------
create table rotation_rates (
  id             serial,
  eta            real,

  -- 

  primary key (id),
  unique (eta),
  constraint positive_eta check (eta >= 0)

);

-----------------------------------------------------------
create table rotation_frequencies (
  id              serial,
  id_model        int not null,
  id_rot          smallint not null,
  freq_crit       real not null,     -- Roche critical
  freq_rot        real not null,

  primary key (id),
  -- foreign key (id_model) references models (id),
  -- foreign key (id_rot) references rotation_rates (id),

  constraint positive_rot_crit check (freq_crit >= 0),
  constraint positive_rot_freq check (freq_rot >= 0)

);

-----------------------------------------------------------
create table mode_types(

  id            serial,
  l             int not null,
  m             int not null,

  primary key (id),

  unique (l, m),
  constraint positive_l check (l >= 0),
  constraint bounded_m  check (m >= -l and m <= l)

);

-----------------------------------------------------------
create unlogged table modes (

  id             bigserial, -- should be huge integer
  id_model       int not null,
  id_rot         smallint not null,
  id_type        smallint not null,
  n_p            int not null,
  n_g            int not null,
  n              int not null,     -- n_pg
  freq           real not null,    -- in inertial frame

  --

  primary key (id),
  -- foreign key (id_model) references models (id),
  -- foreign key (id_rot)   references rotation_rates (id),
  -- foreign key (id_type)  references mode_types (id),

  constraint positive_freq check (freq > 0)

);

-- create index index_modes_id_model on modes (id_model);
-- create index index_modes_id_type on modes (id_type);
-- create index index_freq_n on modes (n, freq);

-----------------------------------------------------------
-- Static Insertions

-----------------------------------------------------------
-- Fill up the rotation_rates table with fixed data

insert into rotation_rates (id, eta)
  values (0, 0);
insert into rotation_rates (id, eta)
  values (1, 5);
insert into rotation_rates (id, eta)
  values (2, 10);
insert into rotation_rates (id, eta)
  values (3, 15);
insert into rotation_rates (id, eta)
  values (4, 20);
insert into rotation_rates (id, eta)
  values (5, 25);
insert into rotation_rates (id, eta)
  values (6, 30);
insert into rotation_rates (id, eta)
  values (7, 35);
insert into rotation_rates (id, eta)
  values (8, 40);
insert into rotation_rates (id, eta)
  values (9, 45);
insert into rotation_rates (id, eta)
  values (10, 50);

-----------------------------------------------------------
-- Fill up the mode_types table with fixed-valued data

insert into mode_types (id, l, m) values (0, 0, 0);
insert into mode_types (id, l, m) values (1, 1, 1);
insert into mode_types (id, l, m) values (2, 1, 0);
insert into mode_types (id, l, m) values (3, 1, -1);
insert into mode_types (id, l, m) values (4, 2, 2);
insert into mode_types (id, l, m) values (5, 2, 1);
insert into mode_types (id, l, m) values (6, 2, 0);
insert into mode_types (id, l, m) values (7, 2, -1);
insert into mode_types (id, l, m) values (8, 2, -2);

-----------------------------------------------------------
-- Views of different tables

-- create view view_tracks as select * from tracks;
-----------------------------------------------------------

-- drop schema asamba cascade;
-- drop table tracks, ages, models, rotation_rates;
-- drop table rotation_rates restrict;
-- drop table models_info cascade;
