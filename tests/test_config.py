"""
Tests for configuration loading and validation functionality.
"""
from grappa import should


def test_load_config_blank_model(mock_config_blank_model):
    """Test configuration loading with blank model"""
    from quorum.oai_proxy import load_config

    config = load_config()

    # Verify the exact structure and values from mock config
    config | should.have.length(2)  # primary_backends and settings only
    config["settings"]["timeout"] | should.equal(30)  # specific timeout value
    config["primary_backends"] | should.have.length(1)
    backend = config["primary_backends"][0]
    backend | should.have.keys("name", "url", "model")
    backend["name"] | should.equal("LLM1")
    backend["url"] | should.equal("http://test.example.com/v1")
    backend["model"] | should.equal("")


def test_load_config_with_model(mock_config_with_model):
    """Test configuration loading with model set"""
    from quorum.oai_proxy import load_config

    config = load_config()

    # Verify the exact structure and values from mock config
    config | should.have.length(2)  # primary_backends and settings only
    config["settings"]["timeout"] | should.equal(30)  # specific timeout value
    config["primary_backends"] | should.have.length(1)
    backend = config["primary_backends"][0]
    backend | should.have.keys("name", "url", "model")
    backend["name"] | should.equal("LLM1")
    backend["url"] | should.equal("http://test.example.com/v1")
    backend["model"] | should.equal("gpt-4-test") 