use directories::ProjectDirs;
use std::{env::current_dir, path::PathBuf};
use topiary::{default_configuration_toml, Configuration};

use crate::error::{CLIResult, TopiaryError};

pub fn parse_configuration() -> CLIResult<Configuration> {
    user_configuration_toml()?
        .try_into()
        .map_err(TopiaryError::from)
}

/// User configured languages.toml file, merged with the default config.
fn user_configuration_toml() -> CLIResult<toml::Value> {
    let config = [find_configuration_dir(), find_workspace().join(".topiary")]
        .into_iter()
        .map(|path| path.join("languages.toml"))
        .filter_map(|file| {
            std::fs::read_to_string(file)
                .map(|config| toml::from_str(&config))
                .ok()
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(default_configuration_toml(), |a, b| {
            merge_toml_values(a, b, 3)
        });

    Ok(config)
}

/// Merge two TOML documents, merging values from `right` onto `left`
///
/// When an array exists in both `left` and `right`, `right`'s array is
/// used. When a table exists in both `left` and `right`, the merged table
/// consists of all keys in `left`'s table unioned with all keys in `right`
/// with the values of `right` being merged recursively onto values of
/// `left`.
///
/// `merge_toplevel_arrays` controls whether a top-level array in the TOML
/// document is merged instead of overridden. This is useful for TOML
/// documents that use a top-level array of values like the `languages.toml`,
/// where one usually wants to override or add to the array instead of
/// replacing it altogether.
///
/// NOTE: This merge function is taken from Helix:
/// https://github.com/helix-editor/helix licensed under MPL-2.0. There
/// it is defined under: helix-loader/src/lib.rs. Taken from commit df09490
pub fn merge_toml_values(left: toml::Value, right: toml::Value, merge_depth: usize) -> toml::Value {
    use toml::Value;

    fn get_name(v: &Value) -> Option<&str> {
        v.get("name").and_then(Value::as_str)
    }

    match (left, right) {
        (Value::Array(mut left_items), Value::Array(right_items)) => {
            // The top-level arrays should be merged but nested arrays should
            // act as overrides. For the `languages.toml` config, this means
            // that you can specify a sub-set of languages in an overriding
            // `languages.toml` but that nested arrays like file extensions
            // arguments are replaced instead of merged.
            if merge_depth > 0 {
                left_items.reserve(right_items.len());
                for rvalue in right_items {
                    let lvalue = get_name(&rvalue)
                        .and_then(|rname| {
                            left_items.iter().position(|v| get_name(v) == Some(rname))
                        })
                        .map(|lpos| left_items.remove(lpos));
                    let mvalue = match lvalue {
                        Some(lvalue) => merge_toml_values(lvalue, rvalue, merge_depth - 1),
                        None => rvalue,
                    };
                    left_items.push(mvalue);
                }
                Value::Array(left_items)
            } else {
                Value::Array(right_items)
            }
        }
        (Value::Table(mut left_map), Value::Table(right_map)) => {
            if merge_depth > 0 {
                for (rname, rvalue) in right_map {
                    match left_map.remove(&rname) {
                        Some(lvalue) => {
                            let merged_value = merge_toml_values(lvalue, rvalue, merge_depth - 1);
                            left_map.insert(rname, merged_value);
                        }
                        None => {
                            left_map.insert(rname, rvalue);
                        }
                    }
                }
                Value::Table(left_map)
            } else {
                Value::Table(right_map)
            }
        }
        // Catch everything else we didn't handle, and use the right value
        (_, value) => value,
    }
}

fn find_configuration_dir() -> PathBuf {
    ProjectDirs::from("", "", "topiary")
        .expect("Could not access the OS's Home directory")
        .config_dir()
        .to_owned()
}

pub fn find_workspace() -> PathBuf {
    let current_dir = current_dir().expect("Could not get current working directory");
    for ancestor in current_dir.ancestors() {
        if ancestor.join(".topiary").exists() {
            return ancestor.to_owned();
        }
    }

    // Default to the current dir if we could not find an ancestor with the .topiary directory
    // If current_dir does not contain a .topiary, it will be filtered our in the `user_lang_toml` function.
    current_dir
}
