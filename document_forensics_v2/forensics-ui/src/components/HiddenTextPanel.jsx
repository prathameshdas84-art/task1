// Hidden Text Recovery panel — shows recovered covered/overwritten text
// (or the all-clear note) above the tab bar.

const fieldIcons = {
    name:       "👤",
    amount:     "💰",
    date:       "📅",
    id_number:  "🔢",
    address:    "📍",
    score:      "📊",
    unknown:    "📄",
  };

const methodBadges = {
    white_rectangle_cover: {
      label: "White-out detected",
      color: "#dc2626",
      icon: "🟥",
    },
    text_overlap: {
      label: "Text layered over original",
      color: "#d97706",
      icon: "📑",
    },
    incremental_update: {
      label: "Previous version found",
      color: "#7c3aed",
      icon: "🕐",
    },
  };

export default function HiddenTextPanel({ hiddenTextData }) {
  if (!hiddenTextData) return null;
  return (
    <>
          {hiddenTextData && hiddenTextData.total_found > 0 && (
            <div style={{
              marginTop: '24px',
              border: '2px solid #ef4444',
              borderRadius: '12px',
              overflow: 'hidden',
              boxShadow: '0 4px 12px rgba(239,68,68,0.15)',
              marginBottom: '16px',
            }}>

              {/* Header */}
              <div style={{
                backgroundColor: '#7f1d1d',
                color: 'white',
                padding: '16px 20px',
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
              }}>
                <span style={{fontSize: '20px'}}>🔍</span>
                <div>
                  <div style={{fontWeight: 'bold', fontSize: '16px'}}>
                    HIDDEN TEXT RECOVERY
                  </div>
                  <div style={{fontSize: '12px', opacity: 0.85}}>
                    Original content found beneath the edits
                  </div>
                </div>
                <div style={{
                  marginLeft: 'auto',
                  backgroundColor: '#ef4444',
                  borderRadius: '20px',
                  padding: '4px 12px',
                  fontSize: '13px',
                  fontWeight: 'bold',
                }}>
                  {hiddenTextData.total_found} region
                  {hiddenTextData.total_found > 1 ? 's' : ''} found
                </div>
              </div>

              {/* Warning banner */}
              <div style={{
                backgroundColor: '#fef2f2',
                borderBottom: '1px solid #fecaca',
                padding: '12px 20px',
                fontSize: '13px',
                color: '#991b1b',
              }}>
                ⚠️ This document contains hidden layers.
                The following original content was found
                beneath the visible text:
              </div>

              {/* Findings */}
              <div style={{padding: '16px 20px'}}>
                {hiddenTextData.findings.map((finding, idx) => (
                  <div key={idx} style={{
                    backgroundColor: '#fff',
                    border: '1px solid #e5e7eb',
                    borderRadius: '8px',
                    marginBottom: '12px',
                    overflow: 'hidden',
                  }}>

                    {/* Finding header */}
                    <div style={{
                      backgroundColor: '#f8fafc',
                      padding: '10px 16px',
                      borderBottom: '1px solid #e5e7eb',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                    }}>
                      <span>
                        {fieldIcons[finding.field_type] || '📄'}
                      </span>
                      <span style={{
                        fontWeight: 'bold',
                        fontSize: '14px',
                      }}>
                        Page {finding.page} — {
                          finding.field_type === 'unknown'
                            ? 'Content'
                            : finding.field_type.charAt(0).toUpperCase()
                              + finding.field_type.slice(1)
                        } Field
                      </span>
                      <span style={{
                        backgroundColor: finding.replacement_type === 'missing' ? '#b45309' : '#7c3aed',
                        color: 'white',
                        fontSize: '11px',
                        fontWeight: 'bold',
                        padding: '2px 8px',
                        borderRadius: '4px',
                        textTransform: 'uppercase',
                        letterSpacing: '0.5px',
                      }}>
                        {finding.replacement_type === 'missing' ? 'Missing Data' : 'Replaced Data'}
                      </span>
                      <span style={{
                        marginLeft: 'auto',
                        backgroundColor: methodBadges[finding.method]?.color || '#6b7280',
                        color: 'white',
                        fontSize: '11px',
                        padding: '2px 8px',
                        borderRadius: '4px',
                      }}>
                        {methodBadges[finding.method]?.icon} {' '}
                        {methodBadges[finding.method]?.label || finding.method}
                      </span>
                    </div>

                    {/* Original vs Replaced */}
                    <div style={{padding: '16px'}}>

                      {/* Side by side comparison */}
                      <div style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '12px',
                        marginBottom: '12px',
                      }}>

                        {/* Original */}
                        <div style={{
                          backgroundColor: '#f0fdf4',
                          border: '1px solid #86efac',
                          borderRadius: '6px',
                          padding: '12px',
                        }}>
                          <div style={{
                            fontSize: '11px',
                            color: '#166534',
                            fontWeight: 'bold',
                            marginBottom: '6px',
                            textTransform: 'uppercase',
                            letterSpacing: '0.5px',
                          }}>
                            ✅ ORIGINAL (hidden in file)
                          </div>
                          <div style={{
                            fontSize: '15px',
                            fontWeight: 'bold',
                            color: '#15803d',
                            wordBreak: 'break-word',
                          }}>
                            {finding.original_text}
                          </div>
                        </div>

                        {/* Right side: what's visible now — either the
                            replacement text, or (for a "missing" finding) an
                            explicit note that nothing was put in its place. */}
                        <div style={{
                          backgroundColor: '#fef2f2',
                          border: '1px solid #fca5a5',
                          borderRadius: '6px',
                          padding: '12px',
                        }}>
                          <div style={{
                            fontSize: '11px',
                            color: '#991b1b',
                            fontWeight: 'bold',
                            marginBottom: '6px',
                            textTransform: 'uppercase',
                            letterSpacing: '0.5px',
                          }}>
                            {finding.replacement_type === 'missing'
                              ? '🚫 REMOVED (nothing put in its place)'
                              : '❌ REPLACED WITH (visible text)'}
                          </div>
                          <div style={{
                            fontSize: finding.replacement_type === 'missing' ? '13px' : '15px',
                            fontWeight: 'bold',
                            color: '#dc2626',
                            fontStyle: finding.replacement_type === 'missing' ? 'italic' : 'normal',
                            wordBreak: 'break-word',
                          }}>
                            {finding.replacement_type === 'missing'
                              ? 'No replacement text — the original was covered/removed with nothing visible in its place.'
                              : (finding.covering_text || 'Unknown')}
                          </div>
                        </div>
                      </div>

                      {/* How it was done */}
                      <div style={{
                        backgroundColor: '#f8fafc',
                        borderRadius: '6px',
                        padding: '10px 12px',
                        fontSize: '13px',
                        color: '#475569',
                        lineHeight: '1.5',
                      }}>
                        <span style={{
                          fontWeight: 'bold',
                          color: '#334155',
                        }}>
                          How it was done:{' '}
                        </span>
                        {finding.plain_explanation}
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              {/* Conclusion */}
              {hiddenTextData.conclusion && (
                <div style={{
                  backgroundColor: '#1e293b',
                  color: 'white',
                  padding: '16px 20px',
                  fontSize: '13px',
                  lineHeight: '1.6',
                }}>
                  <div style={{
                    fontWeight: 'bold',
                    marginBottom: '6px',
                    fontSize: '14px',
                  }}>
                    CONCLUSION
                  </div>
                  {hiddenTextData.conclusion}
                </div>
              )}
            </div>
          )}

          {/* Show message if no hidden text found */}
          {hiddenTextData && hiddenTextData.total_found === 0 && (
            <div style={{
              marginTop: '16px',
              marginBottom: '16px',
              padding: '12px 16px',
              backgroundColor: '#f0fdf4',
              border: '1px solid #86efac',
              borderRadius: '8px',
              fontSize: '13px',
              color: '#166534',
            }}>
              ✅ No hidden text detected — visible content
              appears to be original
            </div>
          )}

    </>
  );
}
