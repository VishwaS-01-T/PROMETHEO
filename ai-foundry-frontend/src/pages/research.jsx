import React, { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'

/* ─── Confidence helpers ────────────────────────────────────────── */
function ConfidenceBadge({ confidence }) {
	if (confidence === null || confidence === undefined) return null
	const pct = (confidence * 100).toFixed(0)
	if (confidence >= 0.75) return (
		<span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-green-100 text-green-700 border border-green-200">
			✓ Verified {pct}%
		</span>
	)
	if (confidence >= 0.5) return (
		<span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-amber-100 text-amber-700 border border-amber-200">
			~ Partially verified {pct}%
		</span>
	)
	return (
		<span className="inline-flex items-center gap-1 text-xs font-semibold px-2.5 py-0.5 rounded-full bg-gray-200 text-gray-600 border border-gray-300">
			? Low confidence {pct}%
		</span>
	)
}

function getConfidenceStyle(conf) {
	if (conf === null || conf === undefined || conf >= 0.75) return {}
	if (conf >= 0.5) return { opacity: 0.65, borderLeft: '3px solid #6b7280' }
	if (conf >= 0.25) return { opacity: 0.45, fontStyle: 'italic', borderLeft: '3px solid #d97706' }
	return { opacity: 0.3, borderLeft: '3px solid #ef4444' }
}

function getConfidencePrefix(conf) {
	if (conf === null || conf === undefined || conf >= 0.5) return null
	if (conf >= 0.25) return <span className="text-amber-600 font-semibold text-xs mr-1">⚠️ Unverified —</span>
	return <span className="text-red-500 font-semibold text-xs mr-1">❌ Disputed —</span>
}
/* ─────────────────────────────────────────────────────────────── */

export default function Research() {
	const loc = useLocation()
	const [researchData, setResearchData] = useState(loc.state?.researchData || null)

	useEffect(() => {
		if (!researchData) {
			try {
				const stored = localStorage.getItem('campaign_research')
				if (stored) {
					const parsed = JSON.parse(stored)
					if (parsed?.researchData) setResearchData(parsed.researchData)
				}
			} catch (e) {
				console.error('Failed loading research from storage', e)
			}
		}
	}, [researchData])

	return (
		<div className="w-full min-h-screen p-6" style={{background:'#FFFFFF'}}>
			<h1 className="text-3xl font-bold mb-6 text-gray-800" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Research Analytics</h1>
			<div className="rounded-2xl p-7 border border-gray-200 shadow-lg" style={{background:'linear-gradient(135deg, #f5f3ff 0%, #ede9fe 50%, #e9d5ff 100%)'}}>
				<div className="flex items-center gap-3 mb-5">
					<h2 className="text-gray-800 text-2xl font-bold" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Research Insights</h2>
					{researchData?.overall_confidence !== undefined && researchData?.overall_confidence !== null && (
						<ConfidenceBadge confidence={researchData.overall_confidence} />
					)}
					{researchData?.validation_rounds > 0 && (
						<span className="text-xs text-gray-500 font-medium">
							({researchData.validation_rounds} validation round{researchData.validation_rounds > 1 ? 's' : ''})
						</span>
					)}
				</div>
				{researchData ? (
					<div className="grid gap-6">
						<div>
							<h3 className="text-purple-700 text-lg font-semibold mb-4" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Audience Persona</h3>
							<div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-4">
								<div className="bg-white/60 backdrop-blur-sm border border-purple-200 rounded-xl p-5 transition-all duration-300 shadow-sm">
									<div className="text-xs font-bold uppercase tracking-wider text-purple-600 mb-2" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Pain Point</div>
									<div className="text-gray-700 text-sm leading-relaxed">{researchData.audience_persona?.pain_point || '—'}</div>
								</div>
								<div className="bg-white/60 backdrop-blur-sm border border-purple-200 rounded-xl p-5 transition-all duration-300 shadow-sm">
									<div className="text-xs font-bold uppercase tracking-wider text-purple-600 mb-2" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Motivation</div>
									<div className="text-gray-700 text-sm leading-relaxed">{researchData.audience_persona?.motivation || '—'}</div>
								</div>
								<div className="bg-white/60 backdrop-blur-sm border border-purple-200 rounded-xl p-5 transition-all duration-300 shadow-sm">
									<div className="text-xs font-bold uppercase tracking-wider text-purple-600 mb-2" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Channel</div>
									<div className="text-gray-700 text-sm leading-relaxed">{researchData.audience_persona?.preferred_channel || '—'}</div>
								</div>
							</div>
						</div>
						<div>
							<h3 className="text-purple-700 text-lg font-semibold mb-4" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Core Messaging</h3>
							<div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-4">
								<div className="bg-gradient-to-br from-purple-100 to-purple-50 border-purple-200 backdrop-blur-sm border rounded-xl p-5 transition-all duration-300 shadow-sm">
									<div className="text-xs font-bold uppercase tracking-wider text-purple-600 mb-2" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Value Prop</div>
									<div className="text-gray-700 text-sm leading-relaxed">{researchData.core_messaging?.value_proposition || '—'}</div>
								</div>
								<div className="bg-white/60 backdrop-blur-sm border border-purple-200 rounded-xl p-5 transition-all duration-300 shadow-sm">
									<div className="text-xs font-bold uppercase tracking-wider text-purple-600 mb-2" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>Tone</div>
									<div className="text-gray-700 text-sm leading-relaxed">{researchData.core_messaging?.tone_of_voice || '—'}</div>
								</div>
								<div className="bg-white/60 backdrop-blur-sm border border-purple-200 rounded-xl p-5 transition-all duration-300 shadow-sm">
									<div className="text-xs font-bold uppercase tracking-wider text-purple-600 mb-2" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>CTA</div>
									<div className="text-gray-700 text-sm leading-relaxed">{researchData.core_messaging?.call_to_action || '—'}</div>
								</div>
							</div>
						</div>

						{/* Jurisdiction Information */}
						{researchData.jurisdiction_info && researchData.jurisdiction_info.department_name !== 'Unknown' && (
							<div>
								<h3 className="text-purple-700 text-lg font-semibold mb-4" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>
									Regulatory Jurisdiction
								</h3>
								<div className="bg-gradient-to-br from-indigo-50 to-purple-50 border border-indigo-200 rounded-xl p-6 shadow-sm">
									<div className="flex items-center gap-3 mb-3">
										<div className="w-10 h-10 rounded-lg bg-indigo-100 flex items-center justify-center">
											<svg className="w-5 h-5 text-indigo-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
												<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
											</svg>
										</div>
										<div>
											<div className="text-gray-800 font-bold text-base" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>
												{researchData.jurisdiction_info.department_url ? (
													<a
														href={researchData.jurisdiction_info.department_url}
														target="_blank"
														rel="noopener noreferrer"
														className="text-indigo-700 hover:text-indigo-900 underline decoration-indigo-300 hover:decoration-indigo-500 transition-colors"
													>
														{researchData.jurisdiction_info.department_name}
													</a>
												) : (
													researchData.jurisdiction_info.department_name
												)}
											</div>
											<div className="text-xs text-indigo-500 font-medium mt-0.5" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 600}}>
												{researchData.jurisdiction_info.jurisdiction_type}
											</div>
										</div>
									</div>
									{researchData.jurisdiction_info.department_url && (
										<div className="text-xs text-gray-500 mt-2 flex items-center gap-1.5">
											<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
												<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
											</svg>
											<a
												href={researchData.jurisdiction_info.department_url}
												target="_blank"
												rel="noopener noreferrer"
												className="text-indigo-400 hover:text-indigo-600 underline transition-colors"
											>
												{researchData.jurisdiction_info.department_url}
											</a>
										</div>
									)}
								</div>
							</div>
						)}

						{/* Registration Procedure Steps */}
						{researchData.registration_procedure && researchData.registration_procedure.length > 0 && (
							<div>
								<div className="flex items-center gap-2 mb-4">
									<h3 className="text-purple-700 text-lg font-semibold" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>
										Registration Procedure
									</h3>
									{researchData.step_confidence && Object.keys(researchData.step_confidence).length > 0 && (
										<ConfidenceBadge confidence={
											Object.values(researchData.step_confidence).reduce((a, b) => a + b, 0) / Object.values(researchData.step_confidence).length
										} />
									)}
								</div>
								<div className="bg-white/60 backdrop-blur-sm border border-purple-200 rounded-xl p-6 shadow-sm">
									<div className="space-y-3">
										{researchData.registration_procedure.map((step, idx) => {
											const stepConf = researchData.step_confidence?.[String(idx)] ?? null
											return (
												<div key={idx} className="flex items-start gap-3 rounded-lg px-2 py-1 transition-all" style={getConfidenceStyle(stepConf)}>
													<div className="flex-shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-purple-500 to-indigo-500 flex items-center justify-center text-white font-bold text-xs shadow-sm">
														{idx + 1}
													</div>
													<div className="flex-1 text-gray-700 text-sm leading-relaxed pt-1" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 500}}>
														{getConfidencePrefix(stepConf)}
														{step}
													</div>
													{stepConf !== null && (
														<span className="flex-shrink-0 text-xs text-gray-400 pt-1.5">{(stepConf * 100).toFixed(0)}%</span>
													)}
												</div>
											)
										})}
									</div>
								</div>
							</div>
						)}

						{/* Required Documents Section */}
						{researchData.required_documents && researchData.required_documents.length > 0 && (
							<div>
								<div className="flex items-center gap-2 mb-4">
									<h3 className="text-purple-700 text-lg font-semibold" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>
										Required Documents & Compliance
									</h3>
									{researchData.document_confidence && Object.keys(researchData.document_confidence).length > 0 && (
										<ConfidenceBadge confidence={
											Object.values(researchData.document_confidence).reduce((a, b) => a + b, 0) / Object.values(researchData.document_confidence).length
										} />
									)}
								</div>
								<div className="grid gap-4">
									{researchData.required_documents.map((doc, idx) => {
										const docConf = researchData.document_confidence?.[doc.document_name] ?? null
										return (
											<div key={idx} className="bg-white/70 backdrop-blur-sm border border-purple-200 rounded-xl p-5 transition-all duration-300 shadow-sm hover:shadow-md" style={getConfidenceStyle(docConf)}>
												<div className="flex items-start gap-3">
													<div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center text-purple-700 font-bold text-sm">
														{idx + 1}
													</div>
													<div className="flex-1">
														<div className="flex items-center gap-2 mb-1">
															{getConfidencePrefix(docConf)}
															<span className="text-gray-800 font-semibold text-sm" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 700}}>
																{doc.document_name}
															</span>
															{docConf !== null && (
																<span className="text-xs text-gray-400">{(docConf * 100).toFixed(0)}%</span>
															)}
														</div>
														<div className="text-xs text-purple-600 font-medium mb-2" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 600}}>
															Issued by: {doc.issuing_authority}
														</div>
														<div className="text-gray-600 text-sm leading-relaxed mb-2">
															{doc.purpose}
														</div>
														<div className="inline-flex items-center gap-1.5 bg-amber-50 border border-amber-200 text-amber-700 text-xs font-medium px-3 py-1 rounded-full">
															<svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
																<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
															</svg>
															{doc.deadline_note}
														</div>
													</div>
												</div>
											</div>
										)
									})}
								</div>
							</div>
						)}

						{/* Validation Mismatches */}
						{researchData.validation_mismatches && researchData.validation_mismatches.length > 0 && !researchData.govt_fallback_only && (
							<div>
								<h3 className="text-amber-700 text-lg font-semibold mb-4" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>
									⚠️ Validation Notes
								</h3>
								<div className="bg-amber-50 border border-amber-200 rounded-xl p-5 shadow-sm">
									<ul className="space-y-2">
										{researchData.validation_mismatches.map((m, i) => (
											<li key={i} className="text-amber-800 text-sm flex items-start gap-2">
												<span className="flex-shrink-0 mt-0.5">•</span>
												<span>{m}</span>
											</li>
										))}
									</ul>
								</div>
							</div>
						)}

						{/* Government Fallback Card */}
						{researchData.govt_fallback_only && researchData.raw_govt_content && (
							<div>
								<div className="bg-gradient-to-br from-orange-50 to-amber-50 border-2 border-amber-300 rounded-xl p-6 shadow-md">
									<div className="flex items-center gap-3 mb-4">
										<div className="w-10 h-10 rounded-lg bg-amber-100 flex items-center justify-center">
											<svg className="w-6 h-6 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
												<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
											</svg>
										</div>
										<div>
											<h3 className="text-amber-800 text-lg font-bold" style={{fontFamily: 'Urbanist, sans-serif', fontWeight: 800}}>
												Official Source — Government Website
											</h3>
											<p className="text-amber-600 text-sm">
												AI verification was inconclusive. Showing information sourced directly from the official government website.
											</p>
										</div>
									</div>
									{researchData.jurisdiction_info?.department_url && (
										<a
											href={researchData.jurisdiction_info.department_url}
											target="_blank"
											rel="noopener noreferrer"
											className="inline-flex items-center gap-2 text-indigo-700 hover:text-indigo-900 text-sm font-semibold mb-4 underline decoration-indigo-300 hover:decoration-indigo-500 transition-colors"
										>
											<svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
												<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
											</svg>
											Visit {researchData.jurisdiction_info.department_name || 'Official Website'}
										</a>
									)}
									<div className="bg-white/80 border border-amber-200 rounded-lg p-4 max-h-96 overflow-y-auto">
										<pre className="text-gray-700 text-sm whitespace-pre-wrap leading-relaxed" style={{fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace', fontSize: '0.8rem'}}>
											{researchData.raw_govt_content}
										</pre>
									</div>
								</div>
							</div>
						)}
					</div>
				) : (
					<div className="text-gray-400 italic py-10 text-center">No research data passed. Return to Foundry and run the Research Agent.</div>
				)}
			</div>
		</div>
	)
}
