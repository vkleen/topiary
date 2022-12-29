//! This file deals with reading and parsing the configuration file. It is also
//! responsible for combining the builtin one with the one provided by the user.
use std::path::PathBuf;

use crate::{language::Language, project_dirs::TOPIARY_DIRS, FormatterError, FormatterResult};

use serde_derive::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Configuration {
    pub language: Vec<Language>,
}

impl Configuration {
    pub fn parse() -> FormatterResult<Self> {
        let config_path: PathBuf = TOPIARY_DIRS.config_dir().join("languages.toml");
        println!("config_path = {:#?}", config_path);
        // TODO: error
        let config_str: String = std::fs::read_to_string(config_path).unwrap();
        let config: Self = toml::from_str(&config_str).unwrap();
        for lang in &config.language {
            println!("LANGUAGE: {}", lang.name);
        }
        Ok(config)
    }

    pub fn find_language_by_extension(&self, extension: &str) -> &Language {
        todo!()
    }

    pub fn find_language_by_name(&self, name: &str) -> &Language {
        self.language.iter().find(|&l| l.name == name).unwrap()
    }
}
