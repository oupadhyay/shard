//! Simplified MCP (Model Context Protocol) integration for Shard
//!
//! This module provides a simplified approach to MCP that focuses on giving AI models
//! structured guidance on how to use Shard's existing tools, rather than reimplementing
//! everything as MCP tools.

use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;

/// Tool usage guidance for AI models
#[derive(Debug, Serialize, Deserialize)]
pub struct ToolGuidance {
    pub name: String,
    pub description: String,
    pub usage_pattern: String,
    pub parameters: Vec<ToolParameter>,
    pub examples: Vec<ToolExample>,
    pub reasoning_hints: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolParameter {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
    pub default_value: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ToolExample {
    pub scenario: String,
    pub reasoning: String,
    pub sequence: Vec<String>,
}

/// MCP-style tool reasoning guidance
pub struct McpToolReasoning;

impl McpToolReasoning {
    /// Generate comprehensive tool usage guidance for AI models
    pub fn generate_tool_guidance() -> Vec<ToolGuidance> {
        vec![
            Self::wikipedia_guidance(),
            Self::weather_guidance(),
            Self::financial_guidance(),
            Self::arxiv_guidance(),
            Self::ocr_guidance(),
            Self::multi_tool_research_guidance(),
            Self::iterative_research_guidance(),
        ]
    }

    /// Get reasoning instructions for AI models
    pub fn get_reasoning_instructions() -> String {
        r#"# Shard Tool Usage Reasoning Guide

You have access to several powerful research and data tools. Here's how to reason about when and how to use them:

## General Reasoning Principles:

1. **Understand the Query Intent**: Analyze what the user is really asking for
2. **Choose Appropriate Tools**: Select tools that best match the information needed
3. **Consider Tool Combinations**: Some queries benefit from multiple tools
4. **Think About Tool Order**: Some tools provide context for others
5. **Evaluate Information Quality**: Assess the reliability and completeness of results

## Tool Decision Framework:

### When to use Wikipedia Research:
- For general knowledge and background information using GENERIC, foundational terms
- To understand broad concepts, fields, and topics (not specific subtopics)
- To get comprehensive overviews from authoritative articles
- Use terms like "quantum computing", "artificial intelligence", "renewable energy"
- AVOID specific subtopics like "quantum computing companies", "AI stocks", "solar manufacturers"

### When to use ArXiv Search:
- For cutting-edge research and academic papers
- To find the latest scientific developments
- When looking for technical, peer-reviewed information
- For literature reviews or research summaries

### When to use Weather Tools:
- For current weather conditions
- When location and time-sensitive data is needed
- For travel planning or outdoor activity decisions

### When to use Financial Tools:
- For stock prices, market data, company information
- Economic indicators and financial analysis
- Investment research and market trends

### When to use OCR Tools:
- To extract text from images or screenshots
- When dealing with PDFs, documents, or visual content
- For digitizing printed or handwritten text

## Multi-Tool Research Strategies:

1. **Foundational → Specific**: Start with GENERIC Wikipedia terms for broad context, then extract specifics for other tools
2. **Current → Historical**: Use current data tools, then research historical context with generic terms
3. **Broad → Narrow**: Begin with foundational Wikipedia research, then focus on specific aspects using other tools
4. **Verification**: Cross-reference information across multiple sources
5. **Wikipedia Strategy**: Always use broad, foundational terms that lead to comprehensive, authoritative articles

## Tool Combination Examples:

- **Quantum Computing Query**: Wikipedia ("quantum computing") → Extract companies → Financial data (ticker symbols)
- **AI Investment Research**: Wikipedia ("artificial intelligence") → ArXiv (latest research) → Financial data (AI company stocks)
- **Travel Planning**: Weather (current conditions) → Wikipedia (destination city name)
- **Technology Analysis**: Wikipedia (broad technology term) → ArXiv (technical details) → Financial data (industry leaders)

**Wikipedia Query Guidelines:**
- ✅ Good: "quantum computing", "artificial intelligence", "renewable energy", "Tokyo"
- ❌ Bad: "quantum computing companies", "AI stocks", "solar manufacturers", "Tokyo restaurants"

Remember: Always explain your tool choices to help users understand your reasoning process.
"#.to_string()
    }

    fn wikipedia_guidance() -> ToolGuidance {
        ToolGuidance {
            name: "Wikipedia Research".to_string(),
            description: "Search and gather comprehensive information from Wikipedia using GENERIC, foundational terms".to_string(),
            usage_pattern: "Use broad, foundational topics rather than specific subtopics. Let iterative research extract details.".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "query".to_string(),
                    param_type: "string".to_string(),
                    description: "Generic, foundational topic (NOT specific subtopics)".to_string(),
                    required: true,
                    default_value: None,
                },
                ToolParameter {
                    name: "max_iterations".to_string(),
                    param_type: "number".to_string(),
                    description: "Maximum research depth (1-4)".to_string(),
                    required: false,
                    default_value: Some("3".to_string()),
                }
            ],
            examples: vec![
                ToolExample {
                    scenario: "User asks about quantum computing companies and stock prices".to_string(),
                    reasoning: "Start with foundational 'quantum computing' topic, not 'quantum computing companies'".to_string(),
                    sequence: vec![
                        "GOOD: Search for 'quantum computing' - broad, authoritative coverage".to_string(),
                        "BAD: Search for 'quantum computing companies' - too specific, limited scope".to_string(),
                        "Extract company details from broad article for follow-up financial queries".to_string(),
                    ],
                },
                ToolExample {
                    scenario: "User asks about AI developments".to_string(),
                    reasoning: "Use generic 'artificial intelligence' rather than specific subtopics".to_string(),
                    sequence: vec![
                        "GOOD: 'artificial intelligence', 'machine learning'".to_string(),
                        "BAD: 'AI stocks', 'machine learning companies', 'AI startups'".to_string(),
                    ],
                },
            ],
            reasoning_hints: vec![
                "ALWAYS use generic, foundational terms for Wikipedia queries".to_string(),
                "AVOID specific subtopics like 'companies', 'stocks', 'manufacturers'".to_string(),
                "Let the iterative system extract specific details from broad articles".to_string(),
                "Good examples: 'quantum computing', 'renewable energy', 'artificial intelligence'".to_string(),
                "Bad examples: 'quantum computing companies', 'solar stocks', 'AI startups'".to_string(),
                "Broad articles contain more comprehensive, authoritative information".to_string(),
            ],
        }
    }

    fn weather_guidance() -> ToolGuidance {
        ToolGuidance {
            name: "Weather Lookup".to_string(),
            description: "Get current weather conditions for any location worldwide".to_string(),
            usage_pattern: "Use when current weather information is needed for decision making"
                .to_string(),
            parameters: vec![ToolParameter {
                name: "location".to_string(),
                param_type: "string".to_string(),
                description: "City, address, or geographic location".to_string(),
                required: true,
                default_value: None,
            }],
            examples: vec![ToolExample {
                scenario: "User planning outdoor activities".to_string(),
                reasoning: "Current weather affects planning decisions".to_string(),
                sequence: vec![
                    "Get weather for the specific location".to_string(),
                    "Consider forecast implications for activities".to_string(),
                ],
            }],
            reasoning_hints: vec![
                "Essential for time-sensitive decisions".to_string(),
                "Consider geographic accuracy of location".to_string(),
                "Useful for travel and activity planning".to_string(),
            ],
        }
    }

    fn financial_guidance() -> ToolGuidance {
        ToolGuidance {
            name: "Financial Data Lookup".to_string(),
            description: "Retrieve stock prices, market data, and company financial information"
                .to_string(),
            usage_pattern:
                "Use for investment research, market analysis, and financial decision making"
                    .to_string(),
            parameters: vec![ToolParameter {
                name: "query".to_string(),
                param_type: "string".to_string(),
                description: "Stock symbol, company name, or financial instrument".to_string(),
                required: true,
                default_value: None,
            }],
            examples: vec![ToolExample {
                scenario: "User asks about a company's stock performance".to_string(),
                reasoning: "Need current market data for informed discussion".to_string(),
                sequence: vec![
                    "Look up the company's stock symbol".to_string(),
                    "Analyze current price and trends".to_string(),
                    "Consider additional context from news or research".to_string(),
                ],
            }],
            reasoning_hints: vec![
                "Critical for investment and market discussions".to_string(),
                "Combine with other research for full context".to_string(),
                "Consider market timing and volatility".to_string(),
            ],
        }
    }

    fn arxiv_guidance() -> ToolGuidance {
        ToolGuidance {
            name: "ArXiv Research".to_string(),
            description: "Search academic papers and research preprints on ArXiv".to_string(),
            usage_pattern:
                "Use for cutting-edge research, academic references, and technical information"
                    .to_string(),
            parameters: vec![
                ToolParameter {
                    name: "query".to_string(),
                    param_type: "string".to_string(),
                    description: "Research topic, keywords, or specific paper search".to_string(),
                    required: true,
                    default_value: None,
                },
                ToolParameter {
                    name: "max_results".to_string(),
                    param_type: "number".to_string(),
                    description: "Maximum number of papers to return (1-20)".to_string(),
                    required: false,
                    default_value: Some("5".to_string()),
                },
            ],
            examples: vec![ToolExample {
                scenario: "User needs latest research on machine learning".to_string(),
                reasoning: "ArXiv contains the most recent academic work".to_string(),
                sequence: vec![
                    "Search for relevant ML papers with appropriate keywords".to_string(),
                    "Review abstracts for relevance and recency".to_string(),
                    "Summarize key findings and trends".to_string(),
                ],
            }],
            reasoning_hints: vec![
                "Best source for cutting-edge research".to_string(),
                "Use specific technical terms for better results".to_string(),
                "Consider paper recency for rapidly evolving fields".to_string(),
                "Good complement to Wikipedia for technical depth".to_string(),
            ],
        }
    }

    /// Guidance specifically for iterative research workflows
    fn iterative_research_guidance() -> ToolGuidance {
        ToolGuidance {
            name: "Iterative Research Workflow".to_string(),
            description: "Framework for conducting deep, iterative research that builds knowledge progressively".to_string(),
            usage_pattern: "Initial query → Tool selection → Result analysis → Follow-up queries → Knowledge synthesis".to_string(),
            parameters: vec![
                ToolParameter {
                    name: "iteration_depth".to_string(),
                    param_type: "integer".to_string(),
                    description: "Maximum number of research iterations (1-4 recommended)".to_string(),
                    required: false,
                    default_value: Some("3".to_string()),
                },
                ToolParameter {
                    name: "breadth_vs_depth".to_string(),
                    param_type: "string".to_string(),
                    description: "Strategy for exploration - broad overview vs deep dive".to_string(),
                    required: false,
                    default_value: Some("balanced".to_string()),
                },
            ],
            examples: vec![
                ToolExample {
                    scenario: "\"How do neural networks work and what are the latest developments?\"".to_string(),
                    reasoning: "Educational query requiring both foundational understanding and cutting-edge information".to_string(),
                    sequence: vec![
                        "Iteration 1: WIKIPEDIA_LOOKUP 'neural networks' → Basic concepts and history".to_string(),
                        "Iteration 2: WIKIPEDIA_LOOKUP 'deep learning' → Advanced architectures".to_string(),
                        "Iteration 3: ARXIV_LOOKUP 'transformer neural networks' → Latest research".to_string(),
                        "Iteration 4: ARXIV_LOOKUP 'attention mechanisms' → Specific breakthroughs".to_string(),
                        "Synthesis: Build complete picture from basics to cutting-edge".to_string(),
                    ],
                },
                ToolExample {
                    scenario: "\"Research Apple's business strategy and financial performance\"".to_string(),
                    reasoning: "Business analysis requiring multiple perspectives and data sources".to_string(),
                    sequence: vec![
                        "Iteration 1: WIKIPEDIA_LOOKUP 'Apple Inc' → Company overview and history".to_string(),
                        "Iteration 2: FINANCIAL_DATA 'AAPL' → Current financial metrics".to_string(),
                        "Iteration 3: WIKIPEDIA_LOOKUP 'Apple business strategy' → Strategic approach".to_string(),
                        "Iteration 4: ARXIV_LOOKUP 'Apple innovation research' → R&D insights".to_string(),
                        "Analysis: Combine historical context, current performance, strategy, and innovation".to_string(),
                    ],
                },
            ],
            reasoning_hints: vec![
                "PROGRESSIVE DEPTH: Start broad, then narrow focus based on initial findings".to_string(),
                "KNOWLEDGE GAPS: Use each iteration to identify and fill specific knowledge gaps".to_string(),
                "TOOL ROTATION: Don't rely on one tool - rotate between sources for comprehensive coverage".to_string(),
                "ITERATION PLANNING: Plan next iteration based on previous results, not original query alone".to_string(),
                "STOPPING CRITERIA: Stop when diminishing returns or when sufficient depth is achieved".to_string(),
                "CONTEXT PRESERVATION: Maintain awareness of how each iteration builds on previous ones".to_string(),
                "ADAPTIVE STRATEGY: Adjust research direction based on discoveries in each iteration".to_string(),
            ],
        }
    }

    fn ocr_guidance() -> ToolGuidance {
        ToolGuidance {
            name: "OCR Screen Capture".to_string(),
            description: "Capture screenshots and extract text using OCR technology".to_string(),
            usage_pattern: "Use when text needs to be extracted from visual content".to_string(),
            parameters: vec![ToolParameter {
                name: "instructions".to_string(),
                param_type: "string".to_string(),
                description: "Optional guidance for what to look for in the captured image"
                    .to_string(),
                required: false,
                default_value: None,
            }],
            examples: vec![ToolExample {
                scenario: "User needs to digitize a document".to_string(),
                reasoning: "OCR can extract text from images or PDFs".to_string(),
                sequence: vec![
                    "Capture the screen or document".to_string(),
                    "Extract text using OCR".to_string(),
                    "Process and analyze the extracted content".to_string(),
                ],
            }],
            reasoning_hints: vec![
                "Essential for processing visual text content".to_string(),
                "Interactive - guides user through capture process".to_string(),
                "Quality depends on image clarity and contrast".to_string(),
            ],
        }
    }

    fn multi_tool_research_guidance() -> ToolGuidance {
        ToolGuidance {
            name: "Multi-Tool Research Strategy".to_string(),
            description: "Advanced framework for combining multiple research tools to gather comprehensive, multi-faceted information"
                .to_string(),
            usage_pattern: "Sequential or parallel tool execution based on query complexity and information dependencies"
                .to_string(),
            parameters: vec![
                ToolParameter {
                    name: "research_strategy".to_string(),
                    param_type: "string".to_string(),
                    description: "Research approach: sequential, parallel, or adaptive".to_string(),
                    required: false,
                    default_value: Some("adaptive".to_string()),
                },
                ToolParameter {
                    name: "priority_weighting".to_string(),
                    param_type: "string".to_string(),
                    description: "Priority for tool execution: foundational_first, specialized_first, or balanced".to_string(),
                    required: false,
                    default_value: Some("foundational_first".to_string()),
                },
            ],
            examples: vec![
                ToolExample {
                    scenario: "\"Research Tesla's AI developments and their market impact\"".to_string(),
                    reasoning: "Multi-faceted business analysis requiring company context, financial data, and technical research".to_string(),
                    sequence: vec![
                        "Priority 1: FINANCIAL_DATA 'TSLA' → Current market performance and investor sentiment".to_string(),
                        "Priority 2: WIKIPEDIA_LOOKUP 'Tesla Inc' → Company background and business context".to_string(),
                        "Priority 3: ARXIV_LOOKUP 'Tesla autonomous driving AI' → Latest technical developments".to_string(),
                        "Priority 4: WIKIPEDIA_LOOKUP 'Tesla Autopilot' → Product development history".to_string(),
                        "Synthesis: Combine financial metrics, company strategy, and technical capabilities".to_string(),
                    ],
                },
                ToolExample {
                    scenario: "\"Plan a trip to Tokyo including weather and cultural information\"".to_string(),
                    reasoning: "Travel planning requires both practical current information and cultural context".to_string(),
                    sequence: vec![
                        "Priority 1: WEATHER_LOOKUP 'Tokyo' → Current conditions for immediate planning".to_string(),
                        "Priority 2: WIKIPEDIA_LOOKUP 'Tokyo' → City overview, districts, and attractions".to_string(),
                        "Priority 3: WIKIPEDIA_LOOKUP 'Tokyo culture' → Cultural norms and etiquette".to_string(),
                        "Integration: Weather-appropriate cultural activities and travel recommendations".to_string(),
                    ],
                },
                ToolExample {
                    scenario: "\"Understand quantum computing advances and investment opportunities\"".to_string(),
                    reasoning: "Technology investment analysis requiring scientific understanding and financial insight".to_string(),
                    sequence: vec![
                        "Priority 1: WIKIPEDIA_LOOKUP 'quantum computing' → Foundational concepts and history".to_string(),
                        "Priority 2: ARXIV_LOOKUP 'quantum computing breakthroughs' → Recent scientific advances".to_string(),
                        "Priority 3: FINANCIAL_DATA 'IBM' → Major quantum computing company performance".to_string(),
                        "Priority 4: FINANCIAL_DATA 'GOOGL' → Google's quantum computing investments".to_string(),
                        "Analysis: Technical feasibility, commercial viability, and investment landscape".to_string(),
                    ],
                },
                ToolExample {
                    scenario: "\"Research renewable energy trends and policy implications\"".to_string(),
                    reasoning: "Policy analysis requiring technical understanding, current research, and economic context".to_string(),
                    sequence: vec![
                        "Priority 1: WIKIPEDIA_LOOKUP 'renewable energy' → Overview of technologies and policies".to_string(),
                        "Priority 2: ARXIV_LOOKUP 'renewable energy efficiency' → Latest technological advances".to_string(),
                        "Priority 3: FINANCIAL_DATA 'ENPH' → Solar industry financial health".to_string(),
                        "Priority 4: WIKIPEDIA_LOOKUP 'renewable energy policy' → Government initiatives and regulations".to_string(),
                        "Synthesis: Technology capabilities, market dynamics, and policy framework".to_string(),
                    ],
                },
            ],
            reasoning_hints: vec![
                "FOUNDATIONAL FIRST: Start with Wikipedia for broad context before diving into specifics".to_string(),
                "TOOL COMPLEMENTARITY: Use each tool's unique strengths - Wikipedia for context, ArXiv for cutting-edge research, Financial for market data".to_string(),
                "PRIORITY SEQUENCING: Execute high-priority, time-sensitive tools (Weather, Financial) before background research".to_string(),
                "ADAPTIVE STRATEGY: Adjust subsequent tool selections based on findings from earlier tools".to_string(),
                "INFORMATION TRIANGULATION: Cross-reference information across multiple sources for accuracy".to_string(),
                "CONTEXT PRESERVATION: Maintain awareness of how each tool's results inform the others".to_string(),
                "SYNTHESIS PLANNING: Plan how to integrate findings from multiple tools into a coherent response".to_string(),
                "TRANSPARENCY: Always explain your multi-tool strategy and how each tool contributes to the complete picture".to_string(),
                "EFFICIENCY: Don't use multiple tools for redundant information - each should add unique value".to_string(),
                "STOPPING CRITERIA: Stop adding tools when sufficient breadth and depth have been achieved".to_string(),
            ],
        }
    }

    /// Generate system prompt for AI models that includes tool reasoning guidance
    pub fn generate_system_prompt() -> String {
        let guidance = Self::generate_tool_guidance();
        let instructions = Self::get_reasoning_instructions();

        format!(
            "{}\n\n## Available Tools:\n\n{}\n\n## Tool Selection Guidelines:\n\n{}",
            instructions,
            serde_json::to_string_pretty(&guidance)
                .unwrap_or_else(|_| "Tool guidance unavailable".to_string()),
            r#"
When presented with a query:

1. **Analyze** what information is needed
2. **Select** the most appropriate tool(s)
3. **Explain** your reasoning for tool selection
4. **Execute** tools in logical order
5. **Synthesize** results into a comprehensive response

Always be transparent about:
- Why you chose specific tools
- What each tool contributes to the answer
- Any limitations or gaps in the available information
- How different sources complement each other

This approach ensures users understand your research process and can trust your conclusions.
"#
        )
    }

    /// Get tool capabilities summary for quick reference
    pub fn get_tool_capabilities() -> HashMap<String, serde_json::Value> {
        let mut capabilities = HashMap::new();

        capabilities.insert("wikipedia_research".to_string(), json!({
            "description": "Comprehensive encyclopedia research with iterative search",
            "best_for": ["general knowledge", "background information", "concept explanations"],
            "strengths": ["authoritative sources", "comprehensive coverage", "reliable information"],
            "limitations": ["may not have very recent information", "general rather than specialized"]
        }));

        capabilities.insert(
            "arxiv_research".to_string(),
            json!({
                "description": "Academic paper search and research",
                "best_for": ["cutting-edge research", "scientific papers", "technical information"],
                "strengths": ["latest research", "peer-reviewed content", "technical depth"],
                "limitations": ["highly technical", "may be too specialized for general audiences"]
            }),
        );

        capabilities.insert(
            "weather_lookup".to_string(),
            json!({
                "description": "Current weather conditions worldwide",
                "best_for": ["travel planning", "outdoor activities", "location-based decisions"],
                "strengths": ["real-time data", "global coverage", "detailed conditions"],
                "limitations": ["current conditions only", "not extended forecasts"]
            }),
        );

        capabilities.insert(
            "financial_data".to_string(),
            json!({
                "description": "Stock prices, market data, company information",
                "best_for": ["investment research", "market analysis", "company evaluation"],
                "strengths": ["real-time data", "comprehensive metrics", "historical context"],
                "limitations": ["market hours dependent", "financial focus only"]
            }),
        );

        capabilities.insert("ocr_capture".to_string(), json!({
            "description": "Screen capture and text extraction",
            "best_for": ["document digitization", "image text extraction", "visual content processing"],
            "strengths": ["handles visual content", "interactive capture", "text extraction"],
            "limitations": ["quality dependent on source", "requires user interaction"]
        }));

        capabilities
    }
}

/// Helper function to create reasoning-enhanced system prompt
pub fn create_reasoning_enhanced_prompt(base_prompt: &str) -> String {
    format!(
        "{}\n\n{}\n\nRemember to always explain your reasoning when selecting and using tools. This helps users understand your thought process and builds trust in your research methodology.",
        base_prompt,
        McpToolReasoning::generate_system_prompt()
    )
}

/// Export guidance as JSON for external consumption
pub fn export_tool_guidance() -> Result<String, serde_json::Error> {
    let guidance = McpToolReasoning::generate_tool_guidance();
    let capabilities = McpToolReasoning::get_tool_capabilities();

    let export_data = json!({
        "version": "1.0",
        "tool_guidance": guidance,
        "tool_capabilities": capabilities,
        "reasoning_instructions": McpToolReasoning::get_reasoning_instructions(),
        "usage_examples": {
            "single_tool": "For specific information needs, choose the most appropriate single tool",
            "multi_tool": "For complex research, combine tools strategically",
            "reasoning": "Always explain why you selected specific tools for the query"
        }
    });

    serde_json::to_string_pretty(&export_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guidance_generation() {
        let guidance = McpToolReasoning::generate_tool_guidance();
        assert!(!guidance.is_empty());
        assert!(guidance.len() >= 5); // Should have at least 5 tools
    }

    #[test]
    fn test_system_prompt_generation() {
        let prompt = McpToolReasoning::generate_system_prompt();
        assert!(!prompt.is_empty());
        assert!(prompt.contains("Wikipedia"));
        assert!(prompt.contains("ArXiv"));
        assert!(prompt.contains("reasoning"));
    }

    #[test]
    fn test_capabilities_export() {
        let capabilities = McpToolReasoning::get_tool_capabilities();
        assert!(!capabilities.is_empty());
        assert!(capabilities.contains_key("wikipedia_research"));
        assert!(capabilities.contains_key("arxiv_research"));
    }

    #[test]
    fn test_json_export() {
        let exported = export_tool_guidance().unwrap();
        assert!(!exported.is_empty());

        // Verify it's valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&exported).unwrap();
        assert!(parsed["tool_guidance"].is_array());
        assert!(parsed["tool_capabilities"].is_object());
    }
}
